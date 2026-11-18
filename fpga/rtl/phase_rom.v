// phase_rom.v - phase table, one 32-bit word per step, loaded with $readmemh.
// Synchronous read (one clock of latency), so Vivado infers a block or
// distributed ROM from the same description.
//
// Word layout (written by scripts/phases_to_mem.py):
//   [31]    last   last step of the table, the sequencer wraps to address 0
//   [30]    ramp   1: linear ramp from the previous code, 0: step
//   [29]    gate   1: fire the gate pulse in this step
//   [28]    mark   1: first step of a layer (drives the scope trigger output)
//   [27:24] layer  layer number, 0..15
//   [23:12] code B 12-bit DAC code, channel B
//   [11:0]  code A 12-bit DAC code, channel A
`timescale 1ns / 1ps
`default_nettype none

module phase_rom #(
    parameter integer ADDR_W   = 8,
    parameter         MEM_FILE = "phase_table.mem"
) (
    input  wire              clk,
    input  wire [ADDR_W-1:0] addr,
    output reg  [31:0]       data
);
    reg [31:0] mem [0:(1 << ADDR_W) - 1];

    integer i;
    initial begin
        // unused entries read as an empty "last" step, so a short table loops
        for (i = 0; i < (1 << ADDR_W); i = i + 1)
            mem[i] = 32'h8000_0000;
        $readmemh(MEM_FILE, mem);
    end

    always @(posedge clk)
        data <= mem[addr];
endmodule

`default_nettype wire
