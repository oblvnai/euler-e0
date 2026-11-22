// =============================================================================
// Planck Silicon — Euler E0
// euler_pe.v : Processing Element (PE) of the systolic array
//
// One PE = one INT8 x INT8 multiply-accumulate (MAC) unit.
// Output-stationary dataflow: the PE keeps its own result C[i][j] in `acc`
// while activations flow left->right and weights flow top->bottom.
// This is the same basic idea used in TPU-style matrix engines and in the
// "tensor core" class of units, scaled down to a size we can tape out.
// =============================================================================
`default_nettype none

module euler_pe #(
    parameter ACC_W = 19
) (
    input  wire                     clk,
    input  wire                     rst_n,   // synchronous, active low
    input  wire                     clear,   // clear accumulator + pipeline regs
    input  wire                     en,      // advance one systolic step
    input  wire signed [7:0]        a_in,    // activation from the left
    input  wire signed [7:0]        b_in,    // weight from above
    output reg  signed [7:0]        a_out,   // activation to the right
    output reg  signed [7:0]        b_out,   // weight to the PE below
    output reg  signed [ACC_W-1:0]  acc      // C[i][j]
);

    wire signed [15:0] prod = a_in * b_in;

    always @(posedge clk) begin
        if (!rst_n || clear) begin
            a_out <= 8'sd0;
            b_out <= 8'sd0;
            acc   <= {ACC_W{1'b0}};
        end else if (en) begin
            a_out <= a_in;
            b_out <= b_in;
            acc   <= acc + {{(ACC_W-16){prod[15]}}, prod};
        end
    end

endmodule

`default_nettype wire
