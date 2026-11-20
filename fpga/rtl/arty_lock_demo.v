// arty_lock_demo.v - second top level for the Arty A7: closed-loop phase lock
// (IIR) and gated readout (FIR), with a Pmod DA2 and a Pmod AD1.
//
// Lock. DAC A drives an RC network (the stand-in for a thermo-optic phase
// shifter); ADC channel 0 reads its output. Inside the FPGA that voltage,
// plus an optional slow drift, is taken as an interferometer phase and passed
// through a sine table, which gives the "detector" signal. A PI controller
// (iir_biquad) holds the detector at mid-fringe by moving DAC A. Only the lag
// is physical: the fringe and the drift are emulated.
//
// Readout. ADC channel 1 goes through a 64-tap FIR matched to a pulse shape.
// A gate opens periodically; the FIR output for the last sample inside the
// gate is kept as the value of that shot.
//
//   sw[0]  loop closed (down: DAC A parked at mid-scale)
//   sw[1]  drift on
//   sw[2]  DAC B shows: down, the detector signal; up, the last shot value
//   sw[3]  set-point square wave on (step response)
//   btn0   reset
//   led[0] loop closed   led[1] drift on   led[2] locked   led[3] controller at a rail
//   JA 1..4  Pmod DA2      JD 1..4  Pmod AD1
//   JB 1     gate          JB 2     trig (set-point square wave, always running)
`timescale 1ns / 1ps
`default_nettype none

module arty_lock_demo #(
    parameter         FRINGE_FILE  = "fringe_lut.mem",
    parameter         FIR_FILE     = "fir_coeffs.mem",
    parameter integer SAMPLE_DIV   = 1000,   // clk periods per sample: 100 kS/s
    parameter integer PHASE_SHIFT  = 4,      // phase = (code - 2048) << PHASE_SHIFT, 2^16 per turn
    parameter integer DRIFT_STEP   = 1,      // phase units per sample; triangle of +-1/4 turn
    parameter integer GATE_PERIOD  = 1000,   // samples: 10 ms
    parameter integer GATE_LEN     = 80,     // samples: 0.8 ms
    parameter integer STEP_HALF    = 500,    // samples per half period of the set-point square wave
    parameter integer STEP_AMPL    = 300,    // codes
    parameter integer LOCK_WINDOW  = 64,     // |error| below this counts as locked
    parameter integer WARM_SAMPLES = 1024,   // after reset the DAC waits at mid-scale this long (10 tau)
    parameter signed [17:0] PI_B0  = 18'sd52438,    // from scripts/design_filters.py
    parameter signed [17:0] PI_B1  = -18'sd51916,
    parameter integer LOG2_TAPS    = 6,
    parameter integer SPI_HALF_DIV = 4,
    parameter integer SPI_GAP_DIV  = 8
) (
    input  wire       clk100,
    input  wire       btn0,
    input  wire [3:0] sw,
    output wire [3:0] led,
    output wire       da2_sync_n,
    output wire       da2_dina,
    output wire       da2_dinb,
    output wire       da2_sclk,
    output wire       ad1_cs_n,
    input  wire       ad1_d0,
    input  wire       ad1_d1,
    output wire       ad1_sclk,
    output reg        gate,
    output reg        trig
);
    // per sample: ADC conversion, three pipeline steps, FIR, DAC frame
    localparam integer SPI_CLKS = 33 * SPI_HALF_DIV + SPI_GAP_DIV + 2;
    localparam integer DSP_CLKS = (1 << LOG2_TAPS) + 12;
    // elaboration fails, in any tool, if a sample does not fit in SAMPLE_DIV clocks
    generate if (SAMPLE_DIV <= 2 * SPI_CLKS + DSP_CLKS + 4) begin : g_bad_sample_div
        SAMPLE_DIV_too_short_for_one_sample u_err ();
    end endgenerate
    // Other limits: GATE_PERIOD, STEP_HALF and WARM_SAMPLES between 1 and 65536.

    // ---------------------------------------------------------- inputs ---
    (* ASYNC_REG = "TRUE" *) reg [1:0] rst_ff = 2'b11;
    (* ASYNC_REG = "TRUE" *) reg [3:0] sw_ff0 = 4'd0, sw_ff1 = 4'd0;
    always @(posedge clk100) begin
        rst_ff <= {rst_ff[0], btn0};
        sw_ff0 <= sw;
        sw_ff1 <= sw_ff0;
    end
    wire rst       = rst_ff[1];
    wire loop_on   = sw_ff1[0];
    wire drift_on  = sw_ff1[1];
    wire show_shot = sw_ff1[2];
    wire step_on   = sw_ff1[3];

    // ------------------------------------------------- sample scheduler ---
    localparam integer DIV_W = $clog2(SAMPLE_DIV);
    reg [DIV_W-1:0] div;
    reg             tick;
    reg [15:0]      n_gate;       // index of the current sample inside the gate period
    reg [15:0]      n_step;
    reg             sq;           // set-point square wave
    reg signed [15:0] drift;      // phase units, triangle
    reg             drift_up;
    reg [15:0]      warm_cnt;
    reg             warm;         // the RC network has charged to mid-scale

    localparam signed [15:0] DRIFT_MAX = 16'sd16384;

    always @(posedge clk100) begin
        tick <= 1'b0;
        // scope trigger: sq one clock (10 ns) later, written every cycle so the
        // flip-flop has no hold feedback and fits in the I/O cell (IOB TRUE)
        trig <= sq;
        if (rst) begin
            div      <= {DIV_W{1'b0}};
            n_gate   <= GATE_PERIOD - 1;
            n_step   <= 16'd0;
            sq       <= 1'b0;
            gate     <= 1'b0;
            trig     <= 1'b0;
            drift    <= 16'sd0;
            drift_up <= 1'b1;
        end else if (div == SAMPLE_DIV - 1) begin
            div  <= {DIV_W{1'b0}};
            tick <= 1'b1;
            // gate: high while samples 0 .. GATE_LEN-1 of each period are taken
            n_gate <= (n_gate == GATE_PERIOD - 1) ? 16'd0 : n_gate + 1'b1;
            gate   <= (n_gate == GATE_PERIOD - 1) || (n_gate + 1 < GATE_LEN);
            // set-point square wave, also the scope trigger
            if (n_step == STEP_HALF - 1) begin
                n_step <= 16'd0;
                sq     <= ~sq;
            end else
                n_step <= n_step + 1'b1;
            // drift: triangle between -1/4 and +1/4 of a turn, zero when off
            if (!drift_on || !warm) begin
                drift    <= 16'sd0;
                drift_up <= 1'b1;
            end else if (drift_up) begin
                if (drift + DRIFT_STEP >= DRIFT_MAX) drift_up <= 1'b0;
                drift <= drift + DRIFT_STEP;
            end else begin
                if (drift - DRIFT_STEP <= -DRIFT_MAX) drift_up <= 1'b1;
                drift <= drift - DRIFT_STEP;
            end
        end else
            div <= div + 1'b1;
    end

    // ------------------------------------------------------------- ADC ---
    wire        adc_done;
    wire [11:0] adc0, adc1;
    wire        adc_busy;

    ad1_spi #(.HALF_DIV(SPI_HALF_DIV), .GAP_DIV(SPI_GAP_DIV)) u_adc (
        .clk(clk100), .rst(rst), .start(tick), .busy(adc_busy), .done(adc_done),
        .ch0(adc0), .ch1(adc1),
        .cs_n(ad1_cs_n), .d0(ad1_d0), .d1(ad1_d1), .sclk(ad1_sclk)
    );

    // --------------------------------------- emulated fringe and error ---
    (* rom_style = "block" *) reg [11:0] fringe [0:4095];      // one entry per ADC code
    initial $readmemh(FRINGE_FILE, fringe);

    localparam integer PC_W = $clog2(DSP_CLKS + 1);
    reg [PC_W-1:0]    pc;          // pipeline step after the ADC result, 0 = idle
    reg [15:0]        phase;
    reg [11:0]        det = 12'd2048;  // detector signal: the table's read register, so no reset
    reg signed [12:0] err;
    reg               dsp_start;
    reg               dac_start;
    reg [11:0]        dac_a, dac_b;
    reg signed [12:0] shot;
    reg [7:0]         lock_cnt;
    reg               locked;
    // After a reset the capacitor is discharged: half a turn from the lock point,
    // which is an unstable equilibrium of the periodic fringe. Closing the loop
    // there works (it saturates for about 20 samples and locks), but the DAC waits
    // at mid-scale, the lock point, until the capacitor is there too, so the loop
    // starts without that transient.
    wire loop_run = loop_on & warm;

    wire signed [12:0] v_c      = $signed({1'b0, adc0}) - 13'sd2048;
    wire signed [12:0] setpoint = 13'sd2048 + (step_on ? (sq ? STEP_AMPL : -STEP_AMPL) : 0);
    wire signed [12:0] ctrl;
    wire               ctrl_done, ctrl_sat;
    wire signed [12:0] fir_y;
    wire               fir_done, fir_sat;
    wire signed [13:0] a_sum = 14'sd2048 + ctrl;
    wire signed [13:0] s_sum = 14'sd2048 + shot;
    wire [11:0]        shot_code = s_sum[13] ? 12'd0 : (s_sum > 14'sd4095 ? 12'd4095 : s_sum[11:0]);
    wire [12:0]        err_abs = err[12] ? -err : err;

    always @(posedge clk100) begin
        dsp_start <= 1'b0;
        dac_start <= 1'b0;
        if (rst) begin
            pc       <= {PC_W{1'b0}};
            phase    <= 16'd0;
            err      <= 13'sd0;
            dac_a    <= 12'd2048;
            dac_b    <= 12'd2048;
            shot     <= 13'sd0;
            lock_cnt <= 8'd0;
            locked   <= 1'b0;
            warm_cnt <= 16'd0;
            warm     <= 1'b0;
        end else begin
            if (adc_done) begin
                if (warm_cnt == WARM_SAMPLES - 1) warm <= 1'b1;
                else warm_cnt <= warm_cnt + 1'b1;
            end
            if (adc_done) begin
                phase <= (v_c <<< PHASE_SHIFT) + drift;
                pc    <= 1;
            end else if (pc != 0)
                pc <= (pc == DSP_CLKS) ? {PC_W{1'b0}} : pc + 1'b1;

            if (pc == 1) det <= fringe[phase[15:4]];
            if (pc == 2) begin
                err       <= setpoint - $signed({1'b0, det});
                dsp_start <= 1'b1;
            end
            if (pc == 3) begin              // lock detector, once per sample
                if (err_abs < LOCK_WINDOW) begin
                    if (lock_cnt == 8'd255) locked <= 1'b1;
                    else lock_cnt <= lock_cnt + 1'b1;
                end else begin
                    lock_cnt <= 8'd0;
                    locked   <= 1'b0;
                end
            end
            if (fir_done && n_gate == GATE_LEN - 1)
                shot <= fir_y;                 // last sample inside the gate
            if (pc == DSP_CLKS) begin
                dac_a     <= a_sum[11:0];      // ctrl is clamped to +-2047: no overflow
                dac_b     <= show_shot ? shot_code : det;
                dac_start <= 1'b1;
            end
        end
    end

    // -------------------------------------------------- IIR: PI controller ---
    iir_biquad #(
        .B0(PI_B0), .B1(PI_B1), .B2(18'sd0), .NA1(18'sd32768), .NA2(18'sd0),
        .Y_MAX(13'sd2047), .Y_MIN(-13'sd2047)
    ) u_pi (
        .clk(clk100), .rst(rst), .clear(~loop_run), .start(dsp_start), .x(err),
        .y(ctrl), .done(ctrl_done), .sat(ctrl_sat)
    );

    // ------------------------------------------------- FIR: gated readout ---
    fir_serial #(.LOG2_TAPS(LOG2_TAPS), .COEF_FILE(FIR_FILE)) u_fir (
        .clk(clk100), .rst(rst), .start(dsp_start),
        .x($signed({1'b0, adc1}) - 13'sd2048),
        .y(fir_y), .done(fir_done), .sat(fir_sat)
    );

    // ------------------------------------------------------------- DAC ---
    wire dac_busy, dac_done;
    da2_spi #(.HALF_DIV(SPI_HALF_DIV), .GAP_DIV(SPI_GAP_DIV)) u_dac (
        .clk(clk100), .rst(rst), .start(dac_start), .code_a(dac_a), .code_b(dac_b),
        .busy(dac_busy), .done(dac_done),
        .sync_n(da2_sync_n), .dina(da2_dina), .dinb(da2_dinb), .sclk(da2_sclk)
    );

    assign led = {ctrl_sat, locked & loop_run, drift_on, loop_on};
endmodule

`default_nettype wire
