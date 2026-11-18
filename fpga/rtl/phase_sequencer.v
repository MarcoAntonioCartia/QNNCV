// phase_sequencer.v - plays the phase table step by step.
//
// One step, in ticks (one tick = TICK_DIV clock periods, counted from the
// start of the step):
//
//   |<---- 2^RAMP_LOG2 ---->|<- SETTLE ->|<-- GATE -->|<-- HOLD -->|
//   codes move to the target   codes still   gate high    gate low
//   (step: at once; ramp: one
//    DAC write per tick)
//
// A step always lasts the same number of ticks, with or without ramp or gate,
// so the table period is fixed. Linear ramp: after tick i the code is
// prev + floor(i * (target - prev) / 2^RAMP_LOG2), exact at the last tick.
//
// Layer selection: with play_all = 0 only the steps whose layer field equals
// layer_sel are played; the others are skipped in two clock periods each.
`timescale 1ns / 1ps
`default_nettype none

module phase_sequencer #(
    parameter integer ADDR_W       = 8,
    parameter integer TICK_DIV     = 10000,  // clk periods per tick (100 MHz: 100 us)
    parameter integer RAMP_LOG2    = 5,      // ramp length = 2^RAMP_LOG2 ticks (>= 2)
    parameter integer SETTLE_TICKS = 8,      // these three: 1 .. 65536
    parameter integer GATE_TICKS   = 8,
    parameter integer HOLD_TICKS   = 8
) (
    input  wire              clk,
    input  wire              rst,        // synchronous, active high
    input  wire              play_all,
    input  wire [3:0]        layer_sel,
    // phase table
    output reg  [ADDR_W-1:0] rom_addr,
    input  wire [31:0]       rom_data,
    // DAC driver
    output reg               dac_start,
    output reg  [11:0]       dac_code_a,
    output reg  [11:0]       dac_code_b,
    // to the outside
    output reg               gate,
    output reg               trig,
    output reg  [3:0]        cur_layer,
    output reg               step_start  // one-cycle pulse at the start of a played step
);
    localparam [2:0] S_FETCH  = 3'd0,
                     S_DECODE = 3'd1,
                     S_RAMP   = 3'd2,
                     S_SETTLE = 3'd3,
                     S_GATE   = 3'd4,
                     S_HOLD   = 3'd5;

    localparam integer RAMP_TICKS = 1 << RAMP_LOG2;
    localparam integer DIV_W      = (TICK_DIV <= 2) ? 1 : $clog2(TICK_DIV);
    localparam integer ACC_W      = 12 + RAMP_LOG2;

    wire        w_last  = rom_data[31];
    wire        w_ramp  = rom_data[30];
    wire        w_gate  = rom_data[29];
    wire        w_mark  = rom_data[28];
    wire [3:0]  w_layer = rom_data[27:24];
    wire [11:0] w_b     = rom_data[23:12];
    wire [11:0] w_a     = rom_data[11:0];

    reg [2:0]        state;
    reg [DIV_W-1:0]  div;
    reg [15:0]       ticks;
    reg              last_q, ramp_q, gate_q;
    reg [ACC_W-1:0]  acc_a, acc_b;       // code << RAMP_LOG2
    reg signed [12:0] delta_a, delta_b;  // target - previous code

    wire tick = (div == TICK_DIV - 1);

    // sign-extended increments and next accumulator values
    wire [ACC_W-1:0] inc_a = {{(ACC_W-13){delta_a[12]}}, delta_a};
    wire [ACC_W-1:0] inc_b = {{(ACC_W-13){delta_b[12]}}, delta_b};
    wire [ACC_W-1:0] nxt_a = acc_a + inc_a;
    wire [ACC_W-1:0] nxt_b = acc_b + inc_b;

    always @(posedge clk) begin
        dac_start  <= 1'b0;
        step_start <= 1'b0;
        if (rst) begin
            state      <= S_FETCH;
            rom_addr   <= {ADDR_W{1'b0}};
            div        <= {DIV_W{1'b0}};
            ticks      <= 16'd0;
            last_q     <= 1'b0;
            ramp_q     <= 1'b0;
            gate_q     <= 1'b0;
            acc_a      <= {ACC_W{1'b0}};
            acc_b      <= {ACC_W{1'b0}};
            delta_a    <= 13'sd0;
            delta_b    <= 13'sd0;
            dac_code_a <= 12'd0;
            dac_code_b <= 12'd0;
            gate       <= 1'b0;
            trig       <= 1'b0;
            cur_layer  <= 4'd0;
        end else begin
            // tick divider: free running inside a step, cleared when a step starts
            if (state == S_FETCH || state == S_DECODE || tick)
                div <= {DIV_W{1'b0}};
            else
                div <= div + 1'b1;

            case (state)
                S_FETCH:                       // rom_addr is stable, data arrives next cycle
                    state <= S_DECODE;

                S_DECODE: begin
                    if (!play_all && (w_layer != layer_sel)) begin
                        rom_addr <= w_last ? {ADDR_W{1'b0}} : rom_addr + 1'b1;
                        state    <= S_FETCH;
                    end else begin
                        last_q     <= w_last;
                        ramp_q     <= w_ramp;
                        gate_q     <= w_gate;
                        trig       <= w_mark;
                        cur_layer  <= w_layer;
                        step_start <= 1'b1;
                        ticks      <= 16'd0;
                        delta_a    <= $signed({1'b0, w_a}) - $signed({1'b0, dac_code_a});
                        delta_b    <= $signed({1'b0, w_b}) - $signed({1'b0, dac_code_b});
                        acc_a      <= {dac_code_a, {RAMP_LOG2{1'b0}}};
                        acc_b      <= {dac_code_b, {RAMP_LOG2{1'b0}}};
                        if (!w_ramp) begin         // step: write the target now
                            dac_code_a <= w_a;
                            dac_code_b <= w_b;
                            dac_start  <= 1'b1;
                        end
                        state <= S_RAMP;
                    end
                end

                S_RAMP: if (tick) begin
                    if (ramp_q) begin
                        acc_a      <= nxt_a;
                        acc_b      <= nxt_b;
                        dac_code_a <= nxt_a[ACC_W-1:RAMP_LOG2];
                        dac_code_b <= nxt_b[ACC_W-1:RAMP_LOG2];
                        dac_start  <= 1'b1;
                    end
                    if (ticks == RAMP_TICKS - 1) begin
                        ticks <= 16'd0;
                        state <= S_SETTLE;
                    end else
                        ticks <= ticks + 1'b1;
                end

                S_SETTLE: if (tick) begin
                    if (ticks == SETTLE_TICKS - 1) begin
                        ticks <= 16'd0;
                        gate  <= gate_q;
                        state <= S_GATE;
                    end else
                        ticks <= ticks + 1'b1;
                end

                S_GATE: if (tick) begin
                    if (ticks == GATE_TICKS - 1) begin
                        ticks <= 16'd0;
                        gate  <= 1'b0;
                        state <= S_HOLD;
                    end else
                        ticks <= ticks + 1'b1;
                end

                S_HOLD: if (tick) begin
                    if (ticks == HOLD_TICKS - 1) begin
                        trig     <= 1'b0;
                        rom_addr <= last_q ? {ADDR_W{1'b0}} : rom_addr + 1'b1;
                        state    <= S_FETCH;
                    end else
                        ticks <= ticks + 1'b1;
                end

                default: state <= S_FETCH;
            endcase
        end
    end
endmodule

`default_nettype wire
