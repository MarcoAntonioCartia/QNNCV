// arty_phase_demo.v - top level for the Arty A7 (35T or 100T, same pins).
//
// Plays a table of interferometer phases on the two channels of a Pmod DA2
// and fires a gate pulse (pump / readout window) in every step. Digital
// control and readout layer only: no optics is driven by this design.
//
//   sw[3:0]  layer to play, 0..14; 15 (all up) plays the whole table
//   btn0     reset (the table restarts from its first step)
//   led[3:0] layer being played
//   JA 1..4  Pmod DA2: SYNC, DINA, DINB, SCLK
//   JD 1     gate pulse
//   JD 2     scope trigger: high during the first step of each layer
`timescale 1ns / 1ps
`default_nettype none

module arty_phase_demo #(
    parameter         MEM_FILE     = "phase_table.mem",
    parameter integer ADDR_W       = 8,
    parameter integer TICK_DIV     = 10000,  // 100 us at 100 MHz
    parameter integer RAMP_LOG2    = 5,      // 32 ticks: 3.2 ms
    parameter integer SETTLE_TICKS = 8,      // 0.8 ms
    parameter integer GATE_TICKS   = 8,      // 0.8 ms
    parameter integer HOLD_TICKS   = 8,      // 0.8 ms
    parameter integer SPI_HALF_DIV = 4,      // SCLK = 12.5 MHz
    parameter integer SPI_GAP_DIV  = 8       // SYNC high at least 90 ns between frames
) (
    input  wire       clk100,
    input  wire       btn0,
    input  wire [3:0] sw,
    output wire [3:0] led,
    output wire       da2_sync_n,
    output wire       da2_dina,
    output wire       da2_dinb,
    output wire       da2_sclk,
    output wire       gate,
    output wire       trig
);
    // a DAC frame must fit inside one tick, or ramp writes would be dropped
    localparam integer FRAME_CLKS = 33 * SPI_HALF_DIV + SPI_GAP_DIV + 2;
    generate if (TICK_DIV <= FRAME_CLKS) begin : g_bad_tick_div
        TICK_DIV_must_exceed_the_DAC_frame u_err ();   // elaboration fails, in any tool
    end endgenerate

    // two-flop synchronizers for the button and the switches
    (* ASYNC_REG = "TRUE" *) reg [1:0] rst_ff = 2'b11;
    (* ASYNC_REG = "TRUE" *) reg [3:0] sw_ff0 = 4'd0, sw_ff1 = 4'd0;
    always @(posedge clk100) begin
        rst_ff <= {rst_ff[0], btn0};
        sw_ff0 <= sw;
        sw_ff1 <= sw_ff0;
    end
    wire rst = rst_ff[1];

    wire [ADDR_W-1:0] rom_addr;
    wire [31:0]       rom_data;
    wire              dac_start;
    wire [11:0]       dac_code_a, dac_code_b;
    wire              step_start, spi_busy, spi_done;

    phase_rom #(.ADDR_W(ADDR_W), .MEM_FILE(MEM_FILE)) u_rom (
        .clk(clk100), .addr(rom_addr), .data(rom_data)
    );

    phase_sequencer #(
        .ADDR_W(ADDR_W), .TICK_DIV(TICK_DIV), .RAMP_LOG2(RAMP_LOG2),
        .SETTLE_TICKS(SETTLE_TICKS), .GATE_TICKS(GATE_TICKS), .HOLD_TICKS(HOLD_TICKS)
    ) u_seq (
        .clk(clk100), .rst(rst),
        .play_all(sw_ff1 == 4'hF), .layer_sel(sw_ff1),
        .rom_addr(rom_addr), .rom_data(rom_data),
        .dac_start(dac_start), .dac_code_a(dac_code_a), .dac_code_b(dac_code_b),
        .gate(gate), .trig(trig), .cur_layer(led), .step_start(step_start)
    );

    da2_spi #(.HALF_DIV(SPI_HALF_DIV), .GAP_DIV(SPI_GAP_DIV)) u_spi (
        .clk(clk100), .rst(rst),
        .start(dac_start), .code_a(dac_code_a), .code_b(dac_code_b),
        .busy(spi_busy), .done(spi_done),
        .sync_n(da2_sync_n), .dina(da2_dina), .dinb(da2_dinb), .sclk(da2_sclk)
    );
endmodule

`default_nettype wire
