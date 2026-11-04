// ============================================================================
// lloyd_max_quant.sv  (optimized: sign-folding + global thresholds + pipeline)
// Lloyd-Max quantizer, 128 channels in parallel
// Supports mixed precision: per-channel b_sel (2/3/4/5 bit)
//
// Optimizations over original design:
//
//  1. GLOBAL THRESHOLD SHARING
//     After RHT, all dimensions are i.i.d. ~N(0, sigma^2).  Per the TurboQuant
//     algorithm, the optimal Lloyd-Max codebook is therefore the same for every
//     channel at the same bit-width.  We maintain only two global threshold sets
//     (outlier=5-bit, regular=3-bit) instead of per-channel ROM, reducing the
//     threshold input bus from 128×31×16 = 63,488 bits to just 46×16 = 736 bits.
//
//  2. SIGN-MAGNITUDE FOLDING
//     TurboQuant codebooks are symmetric around 0.  We extract sign(y) as the
//     output MSB and compare |y| against positive-half thresholds only, halving
//     the comparator count per channel.
//       5-bit: 31 comparators -> 15 (COMP_MAX = 2^(B_MAX-1) - 1 = 15)
//       3-bit:  7 comparators ->  3
//     For D=128: saves 128 × 16 = 2048 comparators.
//
//  3. PIPELINE REGISTER between abs+compare (Stage 1) and popcount (Stage 2)
//     Breaks the critical path for >400 MHz / potential 1 GHz target.
//
//  4. ZERO-OVERHEAD SCALING (from rht_butterfly)
//     Thresholds stored in this module's ROM must be pre-multiplied by
//     sqrt(128) = 11.3137 during offline codebook generation, since
//     rht_butterfly outputs y/128 rather than y/sqrt(128).
//     The downstream code sees the correct effective comparison.
//
// Latency: 3 cycles (sign+abs → pipeline reg → popcount+pack → output reg)
// Throughput: 1 token/cycle
// ============================================================================

`timescale 1ns/1ps

module lloyd_max_quant #(
    parameter D        = 128,
    parameter B_MAX    = 5,
    parameter DW       = 16,            // data width INT16
    parameter TW       = 16,            // threshold width
    // After sign folding: positive-half thresholds only
    // COMP_MAX = 2^(B_MAX-1) - 1 = 15  (covers 5-bit; 3-bit uses only 3)
    parameter COMP_MAX = (1 << (B_MAX-1)) - 1
)(
    input  wire                             clk,
    input  wire                             rst_n,

    // Input rotated vector (from rht_butterfly, scaled by 1/128)
    input  wire signed [D*DW-1:0]          y_in,
    input  wire                             y_valid,

    // GLOBAL positive-half thresholds — two sets, shared across all channels.
    // Both sets must be pre-multiplied by sqrt(128) to account for rht_butterfly
    // /128 output scaling.  Offline generation:
    //   thresholds_5bit[k] = lm_threshold_5bit[k] * sqrt(128) / 256  (Q7.8)
    //   thresholds_3bit[k] = lm_threshold_3bit[k] * sqrt(128) / 256  (Q7.8)
    input  wire [COMP_MAX*TW-1:0]           global_thresholds_5bit,  // 15 values
    input  wire [3*TW-1:0]                  global_thresholds_3bit,  //  3 values

    // Per-channel bit-width selection: 2-bit encoding
    //   2'b00=2bit(1 cmp), 2'b01=3bit(3 cmp), 2'b10=4bit(7 cmp), 2'b11=5bit(15 cmp)
    input  wire [D*2-1:0]                   b_sel,

    // Output: B_MAX bits per channel.
    //   [B_MAX-1] = sign bit  (0=positive, 1=negative)
    //   [B_MAX-2:0] = magnitude index (thermometer popcount of |y| vs pos thresholds)
    output reg  [D*B_MAX-1:0]              idx_out,
    output reg                              idx_valid
);

    // -------------------------------------------------------------------------
    // Stage 1: Sign extraction + absolute value + parallel comparators
    // Each channel compares |y_ch| against COMP_MAX positive-half thresholds,
    // masked to the active count determined by b_sel.
    // -------------------------------------------------------------------------
    wire                    sign_s1  [0:D-1];     // 0=pos, 1=neg
    wire signed [DW-1:0]    abs_s1   [0:D-1];     // |y_ch| (saturating)
    wire [COMP_MAX-1:0]     cmp_s1   [0:D-1];     // thermometer (positive half)

    genvar ch, t;
    generate
        for (ch = 0; ch < D; ch++) begin : gen_channel
            wire signed [DW-1:0] y_ch = y_in[(ch+1)*DW-1 : ch*DW];

            // Sign and absolute value (saturate MIN_INT)
            assign sign_s1[ch] = y_ch[DW-1];
            assign abs_s1[ch]  = sign_s1[ch]
                                 ? ((y_ch == {1'b1, {(DW-1){1'b0}}})
                                    ? {1'b0, {(DW-1){1'b1}}}
                                    : -y_ch)
                                 : y_ch;

            // Extract bsel once per channel (avoids 15× redundant decode)
            wire [1:0] bsel_ch = b_sel[(ch+1)*2-1 : ch*2];
            wire [3:0] n_active = (bsel_ch == 2'b11) ? 4'd15 :
                                  (bsel_ch == 2'b10) ? 4'd7  :
                                  (bsel_ch == 2'b01) ? 4'd3  :
                                                       4'd1;
            wire use_5bit = (bsel_ch == 2'b11);

            for (t = 0; t < COMP_MAX; t++) begin : gen_cmp
                wire signed [TW-1:0] thr_5 = global_thresholds_5bit[(t+1)*TW-1 : t*TW];
                wire signed [TW-1:0] thr_3 = (t < 3) ? global_thresholds_3bit[(t+1)*TW-1 : t*TW]
                                                       : {TW{1'b0}};
                wire signed [TW-1:0] thr_sel = use_5bit ? thr_5 : thr_3;
                wire active = (t < n_active);

                assign cmp_s1[ch][t] = active & (abs_s1[ch] >= thr_sel);
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Pipeline register between Stage 1 (compare) and Stage 2 (popcount)
    // This register breaks the critical path and enables high-frequency operation.
    // -------------------------------------------------------------------------
    reg                  sign_s1r [0:D-1];
    reg [COMP_MAX-1:0]   cmp_s1r  [0:D-1];
    reg [D*2-1:0]        b_sel_r;
    reg                  pipe_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_valid <= 1'b0;
            b_sel_r    <= '0;
            for (int c = 0; c < D; c++) begin
                sign_s1r[c] <= '0;
                cmp_s1r[c]  <= '0;
            end
        end else begin
            pipe_valid <= y_valid;
            b_sel_r    <= b_sel;
            for (int c = 0; c < D; c++) begin
                sign_s1r[c] <= sign_s1[c];
                cmp_s1r[c]  <= cmp_s1[c];
            end
        end
    end

    // -------------------------------------------------------------------------
    // Stage 2: Popcount (thermometer -> binary magnitude index)
    // The popcount of the thermometer code directly gives the magnitude bin.
    // Synthesis will infer an adder tree (e.g., CSA reduction) automatically.
    // -------------------------------------------------------------------------
    function automatic [B_MAX-2:0] popcount_mag;
        input [COMP_MAX-1:0] thermo;    // active comparator results
        integer s;
        reg [B_MAX-2:0] cnt;
        begin
            cnt = '0;
            for (s = 0; s < COMP_MAX; s++) begin
                if (thermo[s]) cnt = cnt + 1;
            end
            popcount_mag = cnt;
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            idx_valid <= 1'b0;
            idx_out   <= '0;
        end else begin
            idx_valid <= pipe_valid;
            for (int c = 0; c < D; c++) begin
                // MSB = sign, lower bits = magnitude popcount
                idx_out[(c+1)*B_MAX-1 : c*B_MAX] <=
                    {sign_s1r[c], popcount_mag(cmp_s1r[c])};
            end
        end
    end

endmodule
