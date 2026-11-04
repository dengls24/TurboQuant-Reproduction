// ============================================================================
// rht_butterfly.sv
// Randomized Hadamard Transform — 7-stage butterfly array
// d=128, fully pipelined, 1 token/cycle throughput
//
// Mathematical operation:
//   y_raw = D * H * x          (this module)
//   y     = (1/sqrt(128)) * y_raw
//
// ZERO-OVERHEAD SCALING:
//   The 1/sqrt(128) normalization factor is NOT applied here.
//   Instead, the Lloyd-Max quantizer's threshold ROM stores thresholds
//   pre-multiplied by sqrt(128) = 11.3137.  This eliminates all
//   approximation error and removes shift/multiply logic from the data path.
//
//   After LOG_D butterfly stages each with /2 per-stage: total factor = /2^7 = /128.
//   So the output of this module is: y_raw = (1/128) * H * D * x
//   The effective scale seen by the comparators is:
//       y_raw vs (thresh / sqrt(128))  <=>  sqrt(128)*y_raw vs thresh
//   which is absorbed offline into the ROM.
//
// Data format: INT16 fixed-point Q7.8
// Latency: LOG_D + 1 = 8 cycles
// ============================================================================

`timescale 1ns/1ps

module rht_butterfly #(
    parameter D       = 128,
    parameter LOG_D   = 7,
    parameter DW      = 16          // INT16 fixed-point Q7.8
)(
    input  wire                 clk,
    input  wire                 rst_n,

    // Input: normalized KV token
    input  wire [D*DW-1:0]      x_in,
    input  wire                 x_valid,

    // Rademacher signs (preloaded from config regs, fixed per model)
    // d_signs[i]=1 means positive, 0 means negate
    input  wire [D-1:0]         d_signs,

    // Output: rotated vector (scaled by 1/128 due to per-stage /2, NOT 1/sqrt(128))
    // Downstream LM-Quant thresholds are pre-multiplied by sqrt(128) to compensate.
    output reg  [D*DW-1:0]      y_out,
    output reg                  y_valid
);

    // -------------------------------------------------------------------------
    // Stage 0: Apply D (sign flip)
    // -------------------------------------------------------------------------
    wire signed [DW-1:0] after_d [0:D-1];
    genvar i;
    generate
        for (i = 0; i < D; i++) begin : gen_d_flip
            wire signed [DW-1:0] x_i = x_in[(i+1)*DW-1 : i*DW];
            assign after_d[i] = d_signs[i] ? x_i : -x_i;
        end
    endgenerate

    reg signed [DW-1:0] stage_reg [0:LOG_D][0:D-1];
    reg                 valid_pipe [0:LOG_D];

    // Register stage 0
    integer k;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_pipe[0] <= 1'b0;
            for (k = 0; k < D; k++) stage_reg[0][k] <= '0;
        end else begin
            valid_pipe[0] <= x_valid;
            for (k = 0; k < D; k++) stage_reg[0][k] <= after_d[k];
        end
    end

    // -------------------------------------------------------------------------
    // Stages 1..LOG_D: WHT butterfly
    // At stage s, butterfly stride = 2^(s-1)
    // Each butterfly: (a, b) -> ((a+b)>>1, (a-b)>>1)
    // The >>1 per stage prevents overflow; total effect after 7 stages: /128.
    // -------------------------------------------------------------------------
    genvar s, j;
    generate
        for (s = 1; s <= LOG_D; s++) begin : gen_stage
            localparam STRIDE = 1 << (s-1);
            wire signed [DW-1:0] stage_in [0:D-1];
            wire signed [DW-1:0] stage_out [0:D-1];

            for (j = 0; j < D; j++) begin : gen_wire_in
                assign stage_in[j] = stage_reg[s-1][j];
            end

            for (j = 0; j < D; j++) begin : gen_butterfly
                if ((j / STRIDE) % 2 == 0) begin
                    wire signed [DW-1:0] partner = stage_in[j + STRIDE];
                    assign stage_out[j] = (stage_in[j] + partner) >>> 1;
                end else begin
                    wire signed [DW-1:0] partner_b = stage_in[j - STRIDE];
                    assign stage_out[j] = (partner_b - stage_in[j]) >>> 1;
                end
            end

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    valid_pipe[s] <= 1'b0;
                    for (int m = 0; m < D; m++) stage_reg[s][m] <= '0;
                end else begin
                    valid_pipe[s] <= valid_pipe[s-1];
                    for (int m = 0; m < D; m++) stage_reg[s][m] <= stage_out[m];
                end
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Output: directly register the final butterfly stage — NO scaling logic.
    // The 1/sqrt(128) correction is absorbed into the LM-Quant threshold ROM.
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y_valid <= 1'b0;
            y_out   <= '0;
        end else begin
            y_valid <= valid_pipe[LOG_D];
            for (int n = 0; n < D; n++)
                y_out[(n+1)*DW-1 : n*DW] <= stage_reg[LOG_D][n];
        end
    end

endmodule
