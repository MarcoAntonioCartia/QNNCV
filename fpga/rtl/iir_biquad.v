// iir_biquad.v - one second-order IIR section, direct form I, fixed point,
// with a saturating output that is also the value fed back.
//
//   y[n] = B0 x[n] + B1 x[n-1] + B2 x[n-2] + NA1 y[n-1] + NA2 y[n-2]
//
// NA1 and NA2 are the feedback coefficients with the sign already applied
// (NA1 = -a1, NA2 = -a2), so every term adds. The same block is a PI
// controller (NA1 = 1, B2 = NA2 = 0), a PID, a notch or a low-pass; only the
// five numbers change.
//
// Formats: coefficients signed Q3.15 (18 bits), signals signed Q13.12
// (25 bits), so one product fits a single DSP48E1 (25 x 18). The fractional
// bits of y are kept in the state: without them, with this rounding, an
// integrator would ignore errors below 0.5 / (B0 + B1) and the loop would
// settle with an offset.
//
// Saturation: y is clamped to [Y_MIN, Y_MAX] and the clamped value is what
// the recursion stores. For a PI this is anti-windup: the integrator cannot
// run away while the actuator is at a rail.
//
// One multiplier, used five times; the result is registered 8 clock periods
// after the one that takes start. The accumulator starts from the rounding
// constant, so no adder sits between it and the clamp.
`timescale 1ns / 1ps
`default_nettype none

module iir_biquad #(
    parameter signed [17:0] B0  = 18'sd32768,   // Q3.15: 32768 = 1.0
    parameter signed [17:0] B1  = 18'sd0,
    parameter signed [17:0] B2  = 18'sd0,
    parameter signed [17:0] NA1 = 18'sd0,
    parameter signed [17:0] NA2 = 18'sd0,
    parameter signed [12:0] Y_MAX = 13'sd2047,
    parameter signed [12:0] Y_MIN = -13'sd2047
) (
    input  wire               clk,
    input  wire               rst,     // synchronous, active high
    input  wire               clear,   // zero the state (loop open); y reads 0
    input  wire               start,   // one-cycle pulse with x valid
    input  wire signed [12:0] x,
    output wire signed [12:0] y,       // integer part of the output
    output reg                done,    // one-cycle pulse, y updated
    output reg                sat      // the last output was clamped
);
    localparam integer FRAC  = 12;
    localparam integer CFRAC = 15;

    reg  signed [24:0] x0, x1, x2, y1, y2;
    reg  signed [17:0] opa;
    reg  signed [24:0] opb;
    reg  signed [42:0] prod;
    reg  signed [47:0] acc;
    reg  [3:0]         cnt;
    reg                run;

    assign y = y1[24:FRAC];

    // rounded result in Q13.12, before clamping
    wire signed [47:0] res = acc >>> CFRAC;
    wire signed [47:0] hi  = {{23{Y_MAX[12]}}, Y_MAX, {FRAC{1'b0}}};
    wire signed [47:0] lo  = {{23{Y_MIN[12]}}, Y_MIN, {FRAC{1'b0}}};
    wire               over  = (res > hi);
    wire               under = (res < lo);
    wire signed [47:0] clamped = over ? hi : (under ? lo : res);

    always @(posedge clk) begin
        done <= 1'b0;
        prod <= opa * opb;
        if (rst || clear) begin
            x0 <= 25'sd0; x1 <= 25'sd0; x2 <= 25'sd0;
            y1 <= 25'sd0; y2 <= 25'sd0;
            opa <= 18'sd0; opb <= 25'sd0;
            acc <= 48'sd0;
            cnt <= 4'd0;
            run <= 1'b0;
            sat <= 1'b0;
        end else if (!run) begin
            if (start) begin
                x0  <= {x, {FRAC{1'b0}}};
                acc <= 48'sd1 <<< (CFRAC - 1);     // rounding
                cnt <= 4'd0;
                run <= 1'b1;
            end
        end else begin
            cnt <= cnt + 1'b1;
            case (cnt)                       // operands of term cnt
                4'd0: begin opa <= B0;  opb <= x0; end
                4'd1: begin opa <= B1;  opb <= x1; end
                4'd2: begin opa <= B2;  opb <= x2; end
                4'd3: begin opa <= NA1; opb <= y1; end
                4'd4: begin opa <= NA2; opb <= y2; end
                default: ;
            endcase
            if (cnt >= 4'd2 && cnt <= 4'd6)  // product of term cnt - 2
                acc <= acc + prod;
            if (cnt == 4'd7) begin
                y2   <= y1;
                y1   <= clamped[24:0];
                x2   <= x1;
                x1   <= x0;
                sat  <= over | under;
                done <= 1'b1;
                run  <= 1'b0;
            end
        end
    end
endmodule

`default_nettype wire
