// =============================================================================
// Planck Silicon — Euler E0
// project.v : Tiny Tapeout top-level wrapper
//
// Pin map
//   ui_in[7:0]   data in (din)
//   uio_in[2:0]  opcode
//   uio_in[7]    valid (command strobe, sampled every clock)
//   uo_out[7:0]  data out (dout)
//   uio_out[5]   busy  (output)
//   uio_out[6]   done  (output)
//
// Rename the module to tt_um_<your_github_username>_e0 before submission.
// =============================================================================
`default_nettype none

module tt_um_planck_e0 (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);

    parameter N = 2;   // 2x2 array for the first tapeout (see README.md)

    wire busy, done;

    euler_core #(.N(N)) u_core (
        .clk   (clk),
        .rst_n (rst_n),
        .valid (uio_in[7]),
        .op    (uio_in[2:0]),
        .din   (ui_in),
        .dout  (uo_out),
        .busy  (busy),
        .done  (done)
    );

    assign uio_out = {1'b0, done, busy, 5'b0};
    assign uio_oe  = 8'b0110_0000;

    // keep the linter quiet about unused inputs
    wire _unused = &{ena, uio_in[6:3], 1'b0};

endmodule

`default_nettype wire
