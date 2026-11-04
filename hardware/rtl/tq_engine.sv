// ============================================================================
// tq_engine.sv
// TurboQuant Engine — Top-level integration
// Combines: token_preprocessor → rht_butterfly → lloyd_max_quant → qjl_engine
//
// Supports N_PARALLEL=4 token processing in parallel (one per attention head or
// pipelined tokens for the same head)
//
// Write path (quantize):  BF16 token → compressed (idx + norm + qjl_sign)
// Read path (dequantize): compressed → BF16 token (IRHT + centroid lookup)
//
// Target: 400 MHz @ 28nm, < 90 mW, < 0.65 mm²
// ============================================================================

`timescale 1ns/1ps

module tq_engine #(
    parameter D           = 128,
    parameter B_MAX       = 5,
    parameter B_OUTLIER   = 5,
    parameter B_REGULAR   = 3,
    parameter N_OUTLIER   = 32,
    parameter DW          = 16,
    parameter NW          = 16,
    parameter N_PARALLEL  = 4
)(
    input  wire                             clk,
    input  wire                             rst_n,

    // === Write path (quantize, host → flash) ===
    input  wire [N_PARALLEL*D*16-1:0]       kv_tokens_in,   // BF16
    input  wire [N_PARALLEL-1:0]            token_valid_in,
    input  wire [5:0]                       layer_id_in,
    input  wire [6:0]                       head_id_in,

    // Compressed output: norm(16b) + idx(D*B_MAX bits, packed) + qjl_sign(D bits)
    output wire [N_PARALLEL*(NW + D*B_MAX + D)-1:0]  compressed_out,
    output wire [N_PARALLEL-1:0]                      compressed_valid,

    // === Read path (dequantize, flash → host) ===
    input  wire [N_PARALLEL*(NW + D*B_MAX + D)-1:0]  compressed_in,
    input  wire [N_PARALLEL-1:0]                      decomp_valid_in,

    output wire [N_PARALLEL*D*16-1:0]       kv_tokens_out,   // BF16
    output wire [N_PARALLEL-1:0]            tokens_valid_out,

    // === Configuration (APB-lite) ===
    input  wire [31:0]  apb_addr,
    input  wire [31:0]  apb_wdata,
    input  wire         apb_write,
    output reg  [31:0]  apb_rdata,

    // === Monitoring ===
    output reg  [31:0]  tokens_processed_cnt,
    output wire [15:0]  compression_ratio_x100,
    output wire         engine_busy,

    // =========================================================================
    // === Plan B: Logic Die / Hybrid Bonding Interface                      ===
    // === Connects to page_buffer_tq_adapter on the same Logic Die          ===
    // === Active when plan_b_mode = 1; Plan A ports remain connected always ===
    // =========================================================================
    input  wire                             plan_b_mode,      // 1 = Logic Die mode

    // Plan B: compressed tokens from page_buffer_tq_adapter (dequant request)
    input  wire [1024-1:0]                  pb_compressed_in,
    input  wire                             pb_compressed_valid,
    output wire                             pb_compressed_ready,

    // Plan B: dequantized BF16 tokens to page_buffer_tq_adapter
    output wire [1024-1:0]                  pb_decomp_out,
    output wire                             pb_decomp_valid,
    input  wire                             pb_decomp_ready,

    // Plan B: BF16 tokens from page_buffer_tq_adapter (quant request)
    input  wire [N_PARALLEL*D*16-1:0]       pb_token_in,
    input  wire                             pb_token_valid,
    output wire                             pb_token_ready,

    // Plan B: compressed output to page_buffer_tq_adapter
    output wire [1024-1:0]                  pb_compressed_out,
    output wire                             pb_compressed_out_valid,
    input  wire                             pb_compressed_out_ready
);

    // -------------------------------------------------------------------------
    // Configuration registers (loaded at startup via APB or DMA)
    // -------------------------------------------------------------------------
    localparam COMP_MAX = (1 << (B_MAX-1)) - 1;  // = 15 for B_MAX=5

    reg [D-1:0]              d_signs_reg;    // Rademacher signs
    reg [D*2-1:0]            b_sel_reg;      // per-channel bit-width select

    // GLOBAL threshold sets — vastly smaller than per-channel storage.
    // Both sets pre-scaled by sqrt(128) offline (absorbs rht_butterfly /128 factor).
    //   Original: 128 × 31 × 16 = 63,488 bits
    //   Optimized: (15 + 3) × 16 = 288 bits  (288× reduction)
    reg [COMP_MAX*16-1:0]    thresholds_5bit_reg;   // 15 positive-half thresholds
    reg [3*16-1:0]           thresholds_3bit_reg;   //  3 positive-half thresholds

    // b_sel initialization: outlier channels (first N_OUTLIER) get 5-bit
    initial begin
        for (int ch = 0; ch < D; ch++) begin
            if (ch < N_OUTLIER)
                b_sel_reg[ch*2+1 : ch*2] = 2'b11;  // 5-bit
            else
                b_sel_reg[ch*2+1 : ch*2] = 2'b01;  // 3-bit
        end
    end

    // APB write handler
    // Addr map:
    //   0x0_xxxx: d_signs[bit_idx]     (word-addressed, 1-bit payload)
    //   0x1_00xx: b_sel[ch]            (2-bit payload)
    //   0x2_0ttt: thresholds_5bit[t]   (16-bit payload, t in 0..14)
    //   0x3_0ttt: thresholds_3bit[t]   (16-bit payload, t in 0..2)
    always @(posedge clk) begin
        if (apb_write) begin
            case (apb_addr[15:12])
                4'h0: d_signs_reg[apb_addr[6:0]] <= apb_wdata[0];
                4'h1: b_sel_reg[apb_addr[7:0]*2+1 -: 2] <= apb_wdata[1:0];
                4'h2: thresholds_5bit_reg[(apb_addr[3:0]+1)*16-1 -: 16] <= apb_wdata[15:0];
                4'h3: thresholds_3bit_reg[(apb_addr[1:0]+1)*16-1 -: 16] <= apb_wdata[15:0];
                default: ;
            endcase
        end
        apb_rdata <= 32'h0;
    end

    // -------------------------------------------------------------------------
    // Plan B input mux: select between host ports (Plan A) and page buffer (Plan B)
    // -------------------------------------------------------------------------
    wire [N_PARALLEL*D*16-1:0]   eff_tokens_in;
    wire [N_PARALLEL-1:0]        eff_token_valid;

    assign eff_tokens_in   = plan_b_mode ? pb_token_in   : kv_tokens_in;
    assign eff_token_valid = plan_b_mode ? {N_PARALLEL{pb_token_valid}} : token_valid_in;
    assign pb_token_ready  = plan_b_mode;

    // -------------------------------------------------------------------------
    // Parallel instantiation of write path (N_PARALLEL lanes)
    // -------------------------------------------------------------------------
    genvar p;
    generate
        for (p = 0; p < N_PARALLEL; p++) begin : gen_write_lane

            // --- token_preprocessor ---
            wire [D*DW-1:0]  x_norm;
            wire [NW-1:0]    norm_fp;
            wire             norm_valid;

            token_preprocessor #(.D(D), .DW(DW), .NW(NW)) u_prep (
                .clk         (clk),
                .rst_n       (rst_n),
                .token_bf16  (eff_tokens_in[(p+1)*D*16-1 : p*D*16]),
                .token_valid (eff_token_valid[p]),
                .x_norm_out  (x_norm),
                .norm_out    (norm_fp),
                .out_valid   (norm_valid)
            );

            // --- rht_butterfly ---
            wire [D*DW-1:0]  y_rot;
            wire             rot_valid;

            rht_butterfly #(.D(D), .LOG_D(7), .DW(DW)) u_rht (
                .clk     (clk),
                .rst_n   (rst_n),
                .x_in    (x_norm),
                .x_valid (norm_valid),
                .d_signs (d_signs_reg),
                .y_out   (y_rot),
                .y_valid (rot_valid)
            );

            // --- lloyd_max_quant ---
            wire [D*B_MAX-1:0]  idx_q;
            wire                idx_valid;

            lloyd_max_quant #(.D(D), .B_MAX(B_MAX), .DW(DW), .TW(16)) u_lmq (
                .clk                      (clk),
                .rst_n                    (rst_n),
                .y_in                     (y_rot),
                .y_valid                  (rot_valid),
                .global_thresholds_5bit   (thresholds_5bit_reg),
                .global_thresholds_3bit   (thresholds_3bit_reg),
                .b_sel                    (b_sel_reg),
                .idx_out                  (idx_q),
                .idx_valid                (idx_valid)
            );

            // --- Residual computation (for QJL) ---
            // x_hat from MSE quantization (centroid lookup + IRHT)
            // Simplified: feed y_rot directly to QJL for prototype
            // TODO: add centroid lookup + IRHT for x_mse reconstruction
            wire [D*DW-1:0]  residual;
            assign residual = x_norm;  // placeholder; replace with x_norm - x_mse

            wire [$clog2(D/16)-1:0]  s_block_req;
            wire [16*D*4-1:0]        s_block;     // from SRAM (external)

            // --- qjl_engine ---
            wire [D-1:0]  sign_q;
            wire          sign_valid;

            qjl_engine #(.D(D), .S_ROWS(16), .DW(DW), .SW(4)) u_qjl (
                .clk          (clk),
                .rst_n        (rst_n),
                .r_in         (residual),
                .r_valid      (rot_valid),    // trigger with same valid
                .s_block      (s_block),
                .s_block_req  (s_block_req),
                .sign_out     (sign_q),
                .sign_valid   (sign_valid)
            );

            // --- Output pack: norm + idx + sign ---
            // Wait for both idx_valid and sign_valid (align with FIFO in real design)
            // Simplified: use sign_valid as master (QJL is the slowest path: 8 cycles)
            assign compressed_out[(p+1)*(NW+D*B_MAX+D)-1 : p*(NW+D*B_MAX+D)] =
                {sign_q, idx_q, norm_fp};
            assign compressed_valid[p] = sign_valid;
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Read path (dequantize) — instantiated separately
    // -------------------------------------------------------------------------
    // TODO: dequant_engine instantiation
    // Placeholder: pass through zeros
    assign kv_tokens_out    = '0;
    assign tokens_valid_out = '0;

    // -------------------------------------------------------------------------
    // Compression ratio: (D * 16) / effective_bits
    // 3.5-bit: 128*16/128/3.5 = 4.57x → x100 = 457
    // -------------------------------------------------------------------------
    assign compression_ratio_x100 = 16'd457;  // fixed for 3.5-bit config

    // -------------------------------------------------------------------------
    // Token counter & busy signal
    // -------------------------------------------------------------------------
    assign engine_busy = |token_valid_in | |decomp_valid_in
                       | (plan_b_mode & (pb_token_valid | pb_compressed_valid));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tokens_processed_cnt <= '0;
        end else begin
            tokens_processed_cnt <= tokens_processed_cnt +
                $countones(compressed_valid);
        end
    end

    // =========================================================================
    // Plan B: Page Buffer ↔ TQ Engine Mux Logic
    //
    // When plan_b_mode=1, the TQ Engine's write/read paths are driven by the
    // page_buffer_tq_adapter instead of the normal host-facing ports.
    //
    // WRITE (quantize) path in Plan B:
    //   pb_token_in -> (same RHT/LMQ/QJL pipeline) -> pb_compressed_out
    //
    // READ (dequantize) path in Plan B:
    //   pb_compressed_in -> (IRHT + centroid lookup) -> pb_decomp_out
    //
    // Implementation note:
    //   The generate block above already instantiates the write pipeline.
    //   Here we mux the inputs and pack outputs for the Plan B interface.
    //   The 1024-bit page buffer bus carries N_PARALLEL tokens packed together:
    //     compressed_out: N_PARALLEL*(NW + D*B_MAX + D) bits = 4*(16+640+128) = 3136b
    //     We truncate to 1024b for the page bus (1/3 of a page; 3 page bus xfers/token-group)
    //   For a real design, a serializer would handle multi-cycle transfer.
    //   Here we expose the first 1024b of compressed_out as pb_compressed_out.
    // =========================================================================

    // Plan B write path: drive TQ input from page buffer adapter
    // (Muxed: when plan_b_mode, override kv_tokens_in with pb_token_in)
    // Note: The actual mux for token_valid_in is done in the generate block via
    // wire aliasing. For synthesizability, we use assign with ternary.

    // Plan B write output: first 1024 bits of compressed_out
    assign pb_compressed_out       = plan_b_mode ? compressed_out[1023:0] : '0;
    assign pb_compressed_out_valid = plan_b_mode ? compressed_valid[0]    : 1'b0;

    // Plan B read path: forward compressed page data to dequant (placeholder)
    // In full implementation, this drives a dequant_engine instantiation
    assign pb_compressed_ready = plan_b_mode;                  // always accept
    assign pb_decomp_out       = plan_b_mode ? kv_tokens_out[1023:0] : '0;
    assign pb_decomp_valid     = plan_b_mode ? tokens_valid_out[0]   : 1'b0;

endmodule
