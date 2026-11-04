// =============================================================================
// page_buffer_tq_adapter.sv
// Plan B: Page Buffer <-> TQ Engine Interface Adapter
//         Deployed on 3D NAND Logic Die (28nm)
//
// Function:
//   WRITE path: Host BF16 KV tokens -> TQ quantize -> compressed INT page data
//   READ  path: Compressed INT page data -> TQ dequantize -> BF16 KV tokens
//
// Interface:
//   - Page side: 1024-bit page buffer bus (matches 16KB page / 64 parallel channels)
//   - Host side: 2048-bit BF16 bus (64 tokens x 16-bit = 1024-bit, 2 phases)
//   - TQ Engine: streams tokens 4 at a time via tq_engine top-level PLAN_B port
//
// Timing: Pipeline depth ~10 cycles; throughput = 4 tokens/cycle
// Area:   ~0.05 mm² @ 28nm (estimated, mostly mux/FIFO logic)
// =============================================================================

`timescale 1ns/1ps

module page_buffer_tq_adapter #(
    parameter D           = 128,    // Token dimension (head_dim)
    parameter N_PARALLEL  = 4,      // Parallel token channels in TQ Engine
    parameter PB_WIDTH    = 1024,   // Page buffer data bus width (bits)
    parameter TOKEN_BITS  = 16,     // BF16 = 16 bits
    parameter COMP_BITS   = 7       // Average compressed bits/dim (3.5-bit avg)
) (
    input  logic                    clk,
    input  logic                    rst_n,

    // ── Mode ──
    // 0 = READ  path: page_data_in  -> (decompress) -> host_data_out
    // 1 = WRITE path: host_data_in  -> (compress)   -> page_data_out
    input  logic                    mode,           // 0=read, 1=write

    // ── Page Buffer Side (to/from Array Die via Hybrid Bonding) ──
    // READ path: data comes from ECC decoder output
    input  logic [PB_WIDTH-1:0]     page_data_in,   // compressed token stream
    input  logic                    page_data_valid, // page data valid
    output logic                    page_data_ready, // backpressure to ECC

    // WRITE path: output compressed data toward Array Die
    output logic [PB_WIDTH-1:0]     page_data_out,  // compressed token stream
    output logic                    page_data_out_valid,
    input  logic                    page_data_out_ready,

    // ── Host Side (to/from Toggle/ONFI interface or SSD Controller) ──
    // WRITE path: BF16 tokens from host
    input  logic [PB_WIDTH-1:0]     host_data_in,   // BF16 token stream (2 cycles per page)
    input  logic                    host_data_valid,
    output logic                    host_data_ready,

    // READ path: BF16 tokens to host
    output logic [PB_WIDTH-1:0]     host_data_out,  // restored BF16 tokens
    output logic                    host_data_out_valid,
    input  logic                    host_data_out_ready,

    // ── Page ID (used for QJL random seed / token group index) ──
    input  logic [31:0]             page_id,

    // ── TQ Engine Streaming Interface (Plan B mode) ──
    // Write path: send BF16 tokens to TQ for quantization
    output logic [N_PARALLEL*TOKEN_BITS*D-1:0] tq_token_in,
    output logic                    tq_token_in_valid,
    input  logic                    tq_token_in_ready,

    // Write path: receive compressed tokens from TQ
    input  logic [PB_WIDTH-1:0]     tq_compressed_out,
    input  logic                    tq_compressed_valid,
    output logic                    tq_compressed_ready,

    // Read path: send compressed tokens to TQ for dequantization
    output logic [PB_WIDTH-1:0]     tq_decomp_in,
    output logic                    tq_decomp_in_valid,
    input  logic                    tq_decomp_in_ready,

    // Read path: receive BF16 tokens from TQ dequantizer
    input  logic [PB_WIDTH-1:0]     tq_decomp_out,
    input  logic                    tq_decomp_out_valid,
    output logic                    tq_decomp_out_ready,

    // ── Config / Status ──
    output logic [31:0]             tokens_processed,
    output logic [31:0]             pages_processed,
    output logic                    busy
);

// ─────────────────────────────────────────────────────────────────────────────
// Internal FIFOs
// Shallow (4-entry) FIFOs to decouple page-buffer timing from TQ pipeline
// ─────────────────────────────────────────────────────────────────────────────

// WRITE path staging: host_data_in -> host_wr_fifo -> TQ -> comp_fifo -> page_data_out
localparam FIFO_DEPTH = 4;
localparam FIFO_PTR   = 2; // log2(FIFO_DEPTH)

// Host-in FIFO (write path, BF16)
logic [PB_WIDTH-1:0]  host_fifo_mem [0:FIFO_DEPTH-1];
logic [FIFO_PTR-1:0]  host_fifo_wr_ptr, host_fifo_rd_ptr;
logic [FIFO_PTR:0]    host_fifo_cnt;
wire                  host_fifo_full  = (host_fifo_cnt == FIFO_DEPTH);
wire                  host_fifo_empty = (host_fifo_cnt == 0);

// Compressed-out FIFO (write path, INT)
logic [PB_WIDTH-1:0]  comp_fifo_mem [0:FIFO_DEPTH-1];
logic [FIFO_PTR-1:0]  comp_fifo_wr_ptr, comp_fifo_rd_ptr;
logic [FIFO_PTR:0]    comp_fifo_cnt;
wire                  comp_fifo_full  = (comp_fifo_cnt == FIFO_DEPTH);
wire                  comp_fifo_empty = (comp_fifo_cnt == 0);

// ─────────────────────────────────────────────────────────────────────────────
// WRITE PATH LOGIC
// ─────────────────────────────────────────────────────────────────────────────

// Accept host BF16 data into staging FIFO
assign host_data_ready = ~host_fifo_full & mode;

// Forward staged BF16 data to TQ Engine
assign tq_token_in       = host_fifo_empty ? '0 : {N_PARALLEL{host_fifo_mem[host_fifo_rd_ptr]}};
assign tq_token_in_valid = ~host_fifo_empty & mode;

// Single always block for host FIFO (avoids multiple-driver synthesis error)
wire host_fifo_push = mode && host_data_valid && !host_fifo_full;
wire host_fifo_pop  = mode && tq_token_in_valid && tq_token_in_ready;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        host_fifo_wr_ptr <= '0;
        host_fifo_rd_ptr <= '0;
        host_fifo_cnt    <= '0;
    end else begin
        if (host_fifo_push) begin
            host_fifo_mem[host_fifo_wr_ptr] <= host_data_in;
            host_fifo_wr_ptr <= host_fifo_wr_ptr + 1;
        end
        if (host_fifo_pop) begin
            host_fifo_rd_ptr <= host_fifo_rd_ptr + 1;
        end
        // Single driver for cnt: handles simultaneous push+pop correctly
        case ({host_fifo_push, host_fifo_pop})
            2'b10:   host_fifo_cnt <= host_fifo_cnt + 1;
            2'b01:   host_fifo_cnt <= host_fifo_cnt - 1;
            default: host_fifo_cnt <= host_fifo_cnt;  // 00 or 11
        endcase
    end
end

// Accept compressed output from TQ Engine
assign tq_compressed_ready = ~comp_fifo_full;

// Output compressed data to page buffer (toward Array Die)
assign page_data_out       = comp_fifo_empty ? '0 : comp_fifo_mem[comp_fifo_rd_ptr];
assign page_data_out_valid = ~comp_fifo_empty & mode;

// Single always block for comp FIFO (avoids multiple-driver synthesis error)
wire comp_fifo_push = tq_compressed_valid && !comp_fifo_full;
wire comp_fifo_pop  = mode && page_data_out_valid && page_data_out_ready;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        comp_fifo_wr_ptr <= '0;
        comp_fifo_rd_ptr <= '0;
        comp_fifo_cnt    <= '0;
    end else begin
        if (comp_fifo_push) begin
            comp_fifo_mem[comp_fifo_wr_ptr] <= tq_compressed_out;
            comp_fifo_wr_ptr <= comp_fifo_wr_ptr + 1;
        end
        if (comp_fifo_pop) begin
            comp_fifo_rd_ptr <= comp_fifo_rd_ptr + 1;
        end
        case ({comp_fifo_push, comp_fifo_pop})
            2'b10:   comp_fifo_cnt <= comp_fifo_cnt + 1;
            2'b01:   comp_fifo_cnt <= comp_fifo_cnt - 1;
            default: comp_fifo_cnt <= comp_fifo_cnt;
        endcase
    end
end

// ─────────────────────────────────────────────────────────────────────────────
// READ PATH LOGIC
// ─────────────────────────────────────────────────────────────────────────────

// Forward compressed page data to TQ dequantizer
assign tq_decomp_in       = page_data_in;
assign tq_decomp_in_valid = page_data_valid & ~mode;
assign page_data_ready    = tq_decomp_in_ready & ~mode;

// Forward dequantized BF16 data to host
assign host_data_out       = tq_decomp_out;
assign host_data_out_valid = tq_decomp_out_valid & ~mode;
assign tq_decomp_out_ready = host_data_out_ready;

// ─────────────────────────────────────────────────────────────────────────────
// Status / Monitoring
// ─────────────────────────────────────────────────────────────────────────────

// tokens_processed: count individual tokens (each PB_WIDTH transfer = N_PARALLEL tokens)
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        tokens_processed <= '0;
        pages_processed  <= '0;
    end else begin
        // Write path: count compressed outputs
        if (mode && page_data_out_valid && page_data_out_ready)
            tokens_processed <= tokens_processed + N_PARALLEL;
        // Read path: count dequantized outputs
        if (!mode && host_data_out_valid && host_data_out_ready)
            tokens_processed <= tokens_processed + N_PARALLEL;
        // Pages: approximate (PB_WIDTH bits / COMP_BITS / D tokens per page)
        if (mode && page_data_out_valid && page_data_out_ready)
            pages_processed <= pages_processed + 1;
    end
end

assign busy = ~host_fifo_empty | ~comp_fifo_empty
            | tq_token_in_valid | tq_compressed_valid;

// ─────────────────────────────────────────────────────────────────────────────
// Assertions (formal / simulation)
// ─────────────────────────────────────────────────────────────────────────────
`ifdef SIMULATION
// No simultaneous read and write path activity when mode is stable
property mode_stable;
    @(posedge clk) disable iff (!rst_n)
    $stable(mode) |-> (host_data_valid |-> mode);
endproperty

// FIFO overflow check
property host_fifo_no_overflow;
    @(posedge clk) disable iff (!rst_n)
    host_fifo_full |-> ~(host_data_valid && mode);
endproperty

assert property (host_fifo_no_overflow)
    else $error("page_buffer_tq_adapter: host_fifo overflow!");
`endif

endmodule
