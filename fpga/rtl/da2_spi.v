// da2_spi.v - write-only serial driver for the Digilent Pmod DA2
// (two TI DAC121S101, 12 bits, sharing SYNC and SCLK, one data line each).
//
// Frame, per DAC (datasheet SNAS265): SYNC low, 16 bits MSB first, sampled by
// the DAC on the FALLING edges of SCLK; the output updates on the 16th
// falling edge. Word = {2 don't-care, PD1, PD0, D11..D0}; PD = 00 is normal
// operation. Both DACs get their word in the same frame, so the two outputs
// change together.
//
// This driver changes the data lines on the rising edge of SCLK and idles
// with SCLK high, so setup and hold around the sampling edge are both
// HALF_DIV clock periods.
//
//   SCLK frequency  = f_clk / (2 * HALF_DIV)      (100 MHz, HALF_DIV 4: 12.5 MHz)
//   frame duration  = 33 * HALF_DIV + GAP_DIV clock periods (SYNC low for 33 * HALF_DIV)
`timescale 1ns / 1ps
`default_nettype none

module da2_spi #(
    parameter integer HALF_DIV = 4,   // clk periods per half SCLK period (>= 1)
    parameter integer GAP_DIV  = 8    // clk periods SYNC stays high after a frame (>= 1)
) (
    input  wire        clk,
    input  wire        rst,       // synchronous, active high
    input  wire        start,     // one-cycle pulse; ignored while busy
    input  wire [11:0] code_a,
    input  wire [11:0] code_b,
    output wire        busy,
    output reg         done,      // one-cycle pulse when the frame (and gap) ends
    output reg         sync_n,    // Pmod pin 1
    output reg         dina,      // Pmod pin 2
    output reg         dinb,      // Pmod pin 3
    output reg         sclk       // Pmod pin 4
);
    localparam [2:0] S_IDLE  = 3'd0,
                     S_LEAD  = 3'd1,   // SYNC low, SCLK still high
                     S_LOW   = 3'd2,   // SCLK low
                     S_HIGH  = 3'd3,   // SCLK high, next bit on the data lines
                     S_TRAIL = 3'd4,   // after the 16th bit, SCLK high, SYNC low
                     S_GAP   = 3'd5;   // SYNC high, minimum time between frames

    localparam integer CNT_MAX = (GAP_DIV > HALF_DIV) ? GAP_DIV : HALF_DIV;
    localparam integer CNT_W   = (CNT_MAX <= 2) ? 1 : $clog2(CNT_MAX);

    reg [2:0]       state;
    reg [CNT_W-1:0] cnt;
    reg [3:0]       bitcnt;
    reg [15:0]      sh_a, sh_b;

    wire half_done = (cnt == HALF_DIV - 1);
    wire gap_done  = (cnt == GAP_DIV - 1);

    assign busy = (state != S_IDLE);

    always @(posedge clk) begin
        done <= 1'b0;
        if (rst) begin
            state  <= S_IDLE;
            cnt    <= {CNT_W{1'b0}};
            bitcnt <= 4'd0;
            sh_a   <= 16'd0;
            sh_b   <= 16'd0;
            sync_n <= 1'b1;
            sclk   <= 1'b1;
            dina   <= 1'b0;
            dinb   <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (start) begin
                        // bit 15 goes out now, the rest waits in the registers
                        sh_a   <= {3'b000, code_a, 1'b0};
                        sh_b   <= {3'b000, code_b, 1'b0};
                        dina   <= 1'b0;
                        dinb   <= 1'b0;
                        sync_n <= 1'b0;
                        bitcnt <= 4'd0;
                        cnt    <= {CNT_W{1'b0}};
                        state  <= S_LEAD;
                    end
                end
                S_LEAD, S_HIGH: begin
                    if (half_done) begin
                        cnt   <= {CNT_W{1'b0}};
                        sclk  <= 1'b0;                // falling edge: the DAC samples
                        state <= S_LOW;
                    end else
                        cnt <= cnt + 1'b1;
                end
                S_LOW: begin
                    if (half_done) begin
                        cnt  <= {CNT_W{1'b0}};
                        sclk <= 1'b1;
                        if (bitcnt == 4'd15)
                            state <= S_TRAIL;
                        else begin
                            bitcnt <= bitcnt + 1'b1;
                            dina   <= sh_a[15];
                            dinb   <= sh_b[15];
                            sh_a   <= {sh_a[14:0], 1'b0};
                            sh_b   <= {sh_b[14:0], 1'b0};
                            state  <= S_HIGH;
                        end
                    end else
                        cnt <= cnt + 1'b1;
                end
                S_TRAIL: begin
                    if (half_done) begin
                        cnt    <= {CNT_W{1'b0}};
                        sync_n <= 1'b1;
                        state  <= S_GAP;
                    end else
                        cnt <= cnt + 1'b1;
                end
                S_GAP: begin
                    if (gap_done) begin
                        done  <= 1'b1;
                        state <= S_IDLE;
                    end else
                        cnt <= cnt + 1'b1;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
