// ============================================================================
// qjl_engine.sv
// Quantized Johnson-Lindenstrauss (QJL) — sign(S*r) computation
// Uses block-streaming approach: S_ROWS rows of S processed per cycle
//
// S matrix: 128×128, INT4 quantized, stored in 64KB SRAM (external)
// Block size: S_ROWS=16 → 8 cycles to process all 128 output signs
// Each cycle: 16 dot products of length 128 (INT16 × INT4)
//
// For tapeout, S can also be generated via LFSR (area tradeoff)
// ============================================================================

`timescale 1ns/1ps

module qjl_engine #(
    parameter D        = 128,
    parameter S_ROWS   = 16,            // rows processed per cycle
    parameter DW       = 16,            // residual data width INT16
    parameter SW       = 4,             // S matrix quantization width INT4
    parameter AW       = 32             // accumulator width (prevent overflow)
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // Input residual vector r (unit-normalized)
    input  wire signed [D*DW-1:0]       r_in,
    input  wire                         r_valid,

    // S matrix block (S_ROWS rows, D columns, INT4 each)
    // Provided by external SRAM; block_idx selects which rows
    input  wire signed [S_ROWS*D*SW-1:0] s_block,
    output wire [$clog2(D/S_ROWS)-1:0]   s_block_req,   // which block to read next

    // Output: 128-bit sign vector
    output reg  [D-1:0]                 sign_out,
    output reg                          sign_valid
);

    localparam N_BLOCKS = D / S_ROWS;     // 8 blocks

    // -------------------------------------------------------------------------
    // State machine: accumulate over S_ROWS blocks
    // -------------------------------------------------------------------------
    reg [$clog2(N_BLOCKS):0]  block_cnt;    // current block (0..7)
    reg signed [D*DW-1:0]     r_buf;        // latched residual
    reg signed [S_ROWS*AW-1:0] accum;       // partial accumulator (S_ROWS outputs)
    reg signed [D*AW-1:0]     result_accum; // full result accumulator

    typedef enum logic [1:0] {IDLE, COMPUTE, OUTPUT} state_t;
    state_t state;

    assign s_block_req = block_cnt[$clog2(N_BLOCKS)-1:0];

    // -------------------------------------------------------------------------
    // Combinatorial: dot product for current block
    // For each of S_ROWS output rows: accumulate r[j] * S[row][j]
    // -------------------------------------------------------------------------
    wire signed [AW-1:0] dot_products [0:S_ROWS-1];
    genvar row, col;
    generate
        for (row = 0; row < S_ROWS; row++) begin : gen_row
            wire signed [AW-1:0] sum;
            // CSA tree for 128 INT16 × INT4 products
            // For synthesis: let DC optimize the adder tree
            assign dot_products[row] = (function_dot(r_buf, s_block, row));
        end
    endgenerate

    // Helper function: dot product of r with row `row` of s_block
    function automatic signed [AW-1:0] function_dot;
        input signed [D*DW-1:0]         r;
        input signed [S_ROWS*D*SW-1:0]  s;
        input integer                    row;
        automatic signed [AW-1:0] acc = 0;
        for (int j = 0; j < D; j++) begin
            automatic signed [DW-1:0] r_j = r[(j+1)*DW-1 : j*DW];
            automatic signed [SW-1:0] s_ij = s[(row*D + j + 1)*SW-1 : (row*D + j)*SW];
            // Sign-extend SW → DW for proper signed multiplication
            acc = acc + ($signed(r_j) * $signed({{(DW-SW){s_ij[SW-1]}}, s_ij}));
        end
        return acc;
    endfunction

    // -------------------------------------------------------------------------
    // Sequential: block accumulation FSM
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= IDLE;
            block_cnt      <= '0;
            r_buf          <= '0;
            accum          <= '0;
            result_accum   <= '0;
            sign_out       <= '0;
            sign_valid     <= 1'b0;
        end else begin
            sign_valid <= 1'b0;

            case (state)
                IDLE: begin
                    if (r_valid) begin
                        r_buf     <= r_in;
                        block_cnt <= '0;
                        result_accum <= '0;
                        state     <= COMPUTE;
                    end
                end

                COMPUTE: begin
                    // Accumulate dot products from current block into result
                    for (int r = 0; r < S_ROWS; r++) begin
                        // Output index = block_cnt * S_ROWS + r
                        automatic integer out_idx = block_cnt * S_ROWS + r;
                        result_accum[(out_idx+1)*AW-1 : out_idx*AW] <=
                            result_accum[(out_idx+1)*AW-1 : out_idx*AW] + dot_products[r];
                    end

                    if (block_cnt == N_BLOCKS - 1) begin
                        block_cnt <= '0;
                        state     <= OUTPUT;
                    end else begin
                        block_cnt <= block_cnt + 1;
                    end
                end

                OUTPUT: begin
                    // Extract signs from accumulators
                    for (int idx = 0; idx < D; idx++) begin
                        sign_out[idx] <= result_accum[(idx+1)*AW-1];  // sign bit
                    end
                    sign_valid <= 1'b1;
                    state      <= IDLE;
                end
            endcase
        end
    end

endmodule
