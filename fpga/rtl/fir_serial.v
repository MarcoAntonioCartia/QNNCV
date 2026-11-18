// fir_serial.v - FIR filter with 2^LOG2_TAPS taps and one multiplier.
//
//   y[n] = sum_k h[k] x[n-k],  k = 0 .. N-1
//
// Coefficients: signed Q1.17 (18 bits), read from COEF_FILE with $readmemh
// (five hex digits per line, two's complement, h[0] first). Samples: signed
// 13 bits. The sum is rounded, shifted by 17 bits and clamped to 13 bits.
// The result is ready N + 3 clock periods after start.
//
// No feedback: the filter is stable whatever the coefficients are and has no
// limit cycles; its memory is exactly N samples.
`timescale 1ns / 1ps
`default_nettype none

module fir_serial #(
    parameter integer LOG2_TAPS = 6,
    parameter         COEF_FILE = "fir_coeffs.mem"
) (
    input  wire               clk,
    input  wire               rst,     // synchronous, active high (the sample memory is not cleared)
    input  wire               start,   // one-cycle pulse with x valid
    input  wire signed [12:0] x,
    output reg  signed [12:0] y,
    output reg                done,    // one-cycle pulse, y updated
    output reg                sat      // the last output was clamped
);
    localparam integer N     = 1 << LOG2_TAPS;
    localparam integer SHIFT = 17;
    localparam integer ACC_W = 31 + LOG2_TAPS;

    reg signed [17:0] coef [0:N-1];
    reg signed [12:0] xbuf [0:N-1];
    integer i;
    initial begin
        for (i = 0; i < N; i = i + 1) begin
            coef[i] = 18'sd0;
            xbuf[i] = 13'sd0;
        end
        $readmemh(COEF_FILE, coef);
    end

    reg [LOG2_TAPS-1:0]  ptr;          // position of the newest sample
    reg [LOG2_TAPS+1:0]  cnt;
    reg                  run;
    reg signed [17:0]    opa;
    reg signed [12:0]    opb;
    reg signed [30:0]    prod;
    reg signed [ACC_W-1:0] acc;

    wire [LOG2_TAPS-1:0] k   = cnt[LOG2_TAPS-1:0];
    wire [LOG2_TAPS-1:0] rd  = ptr - k;
    wire signed [ACC_W-1:0] rnd = acc + ({{(ACC_W-1){1'b0}}, 1'b1} <<< (SHIFT - 1));
    wire signed [ACC_W-1:0] res = rnd >>> SHIFT;
    wire over  = (res >  4095);
    wire under = (res < -4096);

    always @(posedge clk) begin
        done <= 1'b0;
        prod <= opa * opb;
        if (!run && start && !rst)
            xbuf[ptr + 1'b1] <= x;
        if (run && cnt < N) begin
            opa <= coef[k];
            opb <= xbuf[rd];
        end
        if (rst) begin
            ptr <= {LOG2_TAPS{1'b0}};
            cnt <= {(LOG2_TAPS+2){1'b0}};
            run <= 1'b0;
            acc <= {ACC_W{1'b0}};
            y   <= 13'sd0;
            sat <= 1'b0;
        end else if (!run) begin
            if (start) begin
                ptr <= ptr + 1'b1;
                acc <= {ACC_W{1'b0}};
                cnt <= {(LOG2_TAPS+2){1'b0}};
                run <= 1'b1;
            end
        end else begin
            cnt <= cnt + 1'b1;
            if (cnt >= 2 && cnt <= N + 1)        // product of tap cnt - 2
                acc <= acc + prod;
            if (cnt == N + 2) begin
                y    <= over ? 13'sd4095 : (under ? -13'sd4096 : res[12:0]);
                sat  <= over | under;
                done <= 1'b1;
                run  <= 1'b0;
            end
        end
    end
endmodule

`default_nettype wire
