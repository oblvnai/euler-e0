// =============================================================================
// Planck Silicon — Euler E0
// euler_core.v : N x N INT8 systolic matrix-multiply engine with ReLU
//
// Computes  C = A x B   (A: N x N activations, B: N x N weights, INT8 signed)
// optionally followed by ReLU, i.e. one fully-connected neural-network layer.
//
// Byte-wide command interface (one command per clock while `valid` = 1):
//   op 0  NOP
//   op 1  LOAD_A     write din to A buffer (row-major), pointer auto-increments
//   op 2  LOAD_B     write din to B buffer (row-major), pointer auto-increments
//   op 3  RUN        start the matrix multiply (takes 3N-2 clocks, `busy` high)
//   op 4  READ       dout <= next result byte; results row-major C[0][0]..,
//                    3 bytes per result (24-bit two's complement, little-endian)
//   op 5  CLEAR_PTR  reset all pointers and the `done` flag
//   op 6  SET_CFG    cfg <= din   (bit0 = ReLU enable)
//   op 7  READ_ID    dout <= 8'hE0  (chip ID, used for bring-up)
// =============================================================================
`default_nettype none

module euler_core #(
    parameter N = 4
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid,
    input  wire [2:0]  op,
    input  wire [7:0]  din,
    output reg  [7:0]  dout,
    output wire        busy,
    output reg         done
);

    localparam ACC_W  = 17 + $clog2(N);       // N*(-128*-128) fits with margin
    localparam NN     = N * N;
    localparam PTR_W  = (NN > 1) ? $clog2(NN) : 1;
    localparam T_W    = $clog2(3 * N) + 1;
    localparam T_LAST = 3 * N - 3;            // last systolic step index

    // op 0 is NOP: it falls through to the default branch below
    localparam OP_LOAD_A = 3'd1, OP_LOAD_B = 3'd2, OP_RUN = 3'd3,
               OP_READ = 3'd4, OP_CLEAR = 3'd5, OP_CFG = 3'd6, OP_ID = 3'd7;

    localparam [7:0] CHIP_ID = 8'hE0;

    // sized copies of N for width-clean comparisons and index math
    localparam integer     N_I      = N;
    localparam integer     LAST_I   = NN - 1;
    localparam [T_W:0]     N_K      = N_I[T_W:0];
    localparam [PTR_W-1:0] N_P      = N_I[PTR_W-1:0];
    localparam [PTR_W-1:0] LAST_PTR = LAST_I[PTR_W-1:0];

    // ---------------------------------------------------------------- buffers
    reg [8*NN-1:0]   abuf;     // A[i][k] at byte (i*N + k)
    reg [8*NN-1:0]   bbuf;     // B[k][j] at byte (k*N + j)
    reg [PTR_W-1:0]  aptr, bptr, ridx;
    reg [1:0]        rbyte;
    /* verilator lint_off UNUSEDSIGNAL */
    reg [7:0]        cfg;      // only bit 0 (ReLU) is used today; 7:1 reserved
    /* verilator lint_on UNUSEDSIGNAL */
    reg              running;
    reg [T_W-1:0]    t;

    assign busy = running;

    wire relu_en = cfg[0];
    wire start   = valid && (op == OP_RUN) && !running;
    wire step    = running;

    // ------------------------------------------------------- systolic wiring
    // a_bus[i][j] : activation entering PE(i,j) from the left
    // b_bus[i][j] : weight entering PE(i,j) from above
    wire [8*N*N-1:0]     a_bus;
    wire [8*N*N-1:0]     b_bus;
    /* verilator lint_off UNUSEDSIGNAL */
    wire [8*N*N-1:0]     a_fwd;    // last column's and last row's outputs leave the array
    wire [8*N*N-1:0]     b_fwd;
    /* verilator lint_on UNUSEDSIGNAL */
    wire [ACC_W*NN-1:0]  acc_bus;

    genvar gi, gj;
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : g_row
            for (gj = 0; gj < N; gj = gj + 1) begin : g_col
                localparam integer P = gi * N + gj;

                // Left edge: skewed feed of A. Row i receives A[i][k], k = t - i.
                // k is computed one bit wider than t: its top bit set means t < i.
                if (gj == 0) begin : g_a_edge
                    localparam [T_W:0]     ROW      = gi;
                    localparam integer     BASE_I   = gi * N;
                    localparam [PTR_W-1:0] ROW_BASE = BASE_I[PTR_W-1:0];
                    wire [T_W:0]     k      = {1'b0, t} - ROW;
                    wire             in_rng = running && !k[T_W] && (k < N_K);
                    wire [PTR_W-1:0] a_idx  = ROW_BASE + k[PTR_W-1:0];
                    assign a_bus[8*P +: 8] = in_rng ? abuf[{a_idx, 3'b000} +: 8] : 8'd0;
                end else begin : g_a_link
                    assign a_bus[8*P +: 8] = a_fwd[8*(P-1) +: 8];
                end

                // Top edge: skewed feed of B. Column j receives B[k][j], k = t - j.
                if (gi == 0) begin : g_b_edge
                    localparam [T_W:0]     COL   = gj;
                    localparam [PTR_W-1:0] COL_P = gj;
                    wire [T_W:0]     k      = {1'b0, t} - COL;
                    wire             in_rng = running && !k[T_W] && (k < N_K);
                    wire [PTR_W-1:0] b_idx  = k[PTR_W-1:0] * N_P + COL_P;
                    assign b_bus[8*P +: 8] = in_rng ? bbuf[{b_idx, 3'b000} +: 8] : 8'd0;
                end else begin : g_b_link
                    assign b_bus[8*P +: 8] = b_fwd[8*(P-N) +: 8];
                end

                euler_pe #(.ACC_W(ACC_W)) u_pe (
                    .clk   (clk),
                    .rst_n (rst_n),
                    .clear (start),
                    .en    (step),
                    .a_in  (a_bus[8*P +: 8]),
                    .b_in  (b_bus[8*P +: 8]),
                    .a_out (a_fwd[8*P +: 8]),
                    .b_out (b_fwd[8*P +: 8]),
                    .acc   (acc_bus[ACC_W*P +: ACC_W])
                );
            end
        end
    endgenerate

    // --------------------------------------------------------------- readout
    wire signed [ACC_W-1:0] sel_acc = acc_bus[ACC_W*ridx +: ACC_W];
    wire signed [23:0] res24 = (relu_en && sel_acc[ACC_W-1]) ? 24'sd0 :
                               {{(24-ACC_W){sel_acc[ACC_W-1]}}, sel_acc};
    wire [7:0] res_byte = (rbyte == 2'd0) ? res24[7:0]  :
                          (rbyte == 2'd1) ? res24[15:8] : res24[23:16];

    // ------------------------------------------------------------ control FSM
    always @(posedge clk) begin
        if (!rst_n) begin
            abuf    <= {8*NN{1'b0}};
            bbuf    <= {8*NN{1'b0}};
            aptr    <= {PTR_W{1'b0}};
            bptr    <= {PTR_W{1'b0}};
            ridx    <= {PTR_W{1'b0}};
            rbyte   <= 2'd0;
            cfg     <= 8'd0;
            running <= 1'b0;
            done    <= 1'b0;
            t       <= {T_W{1'b0}};
            dout    <= 8'd0;
        end else begin
            // systolic sequencing
            if (running) begin
                if (t == T_LAST[T_W-1:0]) begin
                    running <= 1'b0;
                    done    <= 1'b1;
                end
                t <= t + 1'b1;
            end

            if (valid && !running) begin
                case (op)
                    OP_LOAD_A: begin
                        abuf[8*aptr +: 8] <= din;
                        aptr <= (aptr == LAST_PTR) ? {PTR_W{1'b0}} : aptr + 1'b1;
                    end
                    OP_LOAD_B: begin
                        bbuf[8*bptr +: 8] <= din;
                        bptr <= (bptr == LAST_PTR) ? {PTR_W{1'b0}} : bptr + 1'b1;
                    end
                    OP_RUN: begin
                        running <= 1'b1;
                        done    <= 1'b0;
                        t       <= {T_W{1'b0}};
                        ridx    <= {PTR_W{1'b0}};
                        rbyte   <= 2'd0;
                    end
                    OP_READ: begin
                        dout <= res_byte;
                        if (rbyte == 2'd2) begin
                            rbyte <= 2'd0;
                            ridx  <= (ridx == LAST_PTR) ? {PTR_W{1'b0}} : ridx + 1'b1;
                        end else begin
                            rbyte <= rbyte + 1'b1;
                        end
                    end
                    OP_CLEAR: begin
                        aptr  <= {PTR_W{1'b0}};
                        bptr  <= {PTR_W{1'b0}};
                        ridx  <= {PTR_W{1'b0}};
                        rbyte <= 2'd0;
                        done  <= 1'b0;
                    end
                    OP_CFG:  cfg  <= din;
                    OP_ID:   dout <= CHIP_ID;
                    default: ;
                endcase
            end
        end
    end

endmodule

`default_nettype wire
