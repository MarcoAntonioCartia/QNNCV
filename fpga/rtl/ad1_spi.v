// ad1_spi.v - serial reader for the Digilent Pmod AD1 (two AD7476A, 12 bits,
// sharing CS and SCLK, one data line each).
//
// Conversion, per ADC (AD7476A datasheet): the falling edge of CS samples the
// input and puts the first of four leading zeros on the data line; each of
// the following SCLK falling edges clocks out the next bit, MSB first. The
// host reads every bit on the next falling edge, 16 in total.
//
// The data pins are registered every clock (d0_q, d1_q) and shifted in at the
// clock edge that drives SCLK low, so each bit is read one clock period
// before the ADC is told to change it: with HALF_DIV = 4 at 100 MHz the bit
// has had 70 ns to arrive and is still held by the ADC.
//
//   SCLK frequency = f_clk / (2 * HALF_DIV)   (100 MHz, HALF_DIV 4: 12.5 MHz, max 20 MHz)
//   conversion     = 33 * HALF_DIV + GAP_DIV clock periods
`timescale 1ns / 1ps
`default_nettype none

module ad1_spi #(
    parameter integer HALF_DIV = 4,   // clk periods per half SCLK period (>= 2)
    parameter integer GAP_DIV  = 8    // clk periods CS stays high after a conversion
) (
    input  wire        clk,
    input  wire        rst,       // synchronous, active high
    input  wire        start,     // one-cycle pulse; ignored while busy
    output wire        busy,
    output reg         done,      // one-cycle pulse; ch0 and ch1 are valid from here on
    output reg  [11:0] ch0,
    output reg  [11:0] ch1,
    output reg         cs_n,      // Pmod pin 1
    input  wire        d0,        // Pmod pin 2
    input  wire        d1,        // Pmod pin 3
    output reg         sclk       // Pmod pin 4
);
    localparam [2:0] S_IDLE  = 3'd0,
                     S_HIGH  = 3'd1,   // SCLK high (the first time: lead after CS falls)
                     S_LOW   = 3'd2,
                     S_TRAIL = 3'd3,
                     S_GAP   = 3'd4;

    localparam integer CNT_MAX = (GAP_DIV > HALF_DIV) ? GAP_DIV : HALF_DIV;
    localparam integer CNT_W   = (CNT_MAX <= 2) ? 1 : $clog2(CNT_MAX);

    reg [2:0]       state;
    reg [CNT_W-1:0] cnt;
    reg [4:0]       nfall;       // falling edges produced so far
    reg [15:0]      sh0, sh1;
    reg             d0_q, d1_q;  // input registers (I/O cell flip-flops)

    wire half_done = (cnt == HALF_DIV - 1);

    assign busy = (state != S_IDLE);

    always @(posedge clk) begin
        d0_q <= d0;
        d1_q <= d1;
        done <= 1'b0;
        if (rst) begin
            state <= S_IDLE;
            cnt   <= {CNT_W{1'b0}};
            nfall <= 5'd0;
            sh0   <= 16'd0;
            sh1   <= 16'd0;
            ch0   <= 12'd0;
            ch1   <= 12'd0;
            cs_n  <= 1'b1;
            sclk  <= 1'b1;
        end else begin
            case (state)
                S_IDLE: if (start) begin
                    cs_n  <= 1'b0;                   // the ADC samples its input here
                    nfall <= 5'd0;
                    cnt   <= {CNT_W{1'b0}};
                    state <= S_HIGH;
                end
                S_HIGH: begin
                    if (half_done) begin
                        cnt   <= {CNT_W{1'b0}};
                        sclk  <= 1'b0;               // falling edge: read the current bit
                        sh0   <= {sh0[14:0], d0_q};
                        sh1   <= {sh1[14:0], d1_q};
                        nfall <= nfall + 1'b1;
                        state <= S_LOW;
                    end else
                        cnt <= cnt + 1'b1;
                end
                S_LOW: begin
                    if (half_done) begin
                        cnt   <= {CNT_W{1'b0}};
                        sclk  <= 1'b1;
                        state <= (nfall == 5'd16) ? S_TRAIL : S_HIGH;
                    end else
                        cnt <= cnt + 1'b1;
                end
                S_TRAIL: begin
                    if (half_done) begin
                        cnt   <= {CNT_W{1'b0}};
                        cs_n  <= 1'b1;
                        ch0   <= sh0[11:0];
                        ch1   <= sh1[11:0];
                        state <= S_GAP;
                    end else
                        cnt <= cnt + 1'b1;
                end
                S_GAP: begin
                    if (cnt == GAP_DIV - 1) begin
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
