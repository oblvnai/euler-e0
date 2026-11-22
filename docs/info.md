## How it works

Euler E0 is a small neural-network accelerator from Planck Silicon. It computes one
fully-connected layer in hardware, `C = ReLU(A x B)`, where A (activations) and B
(weights) are 2x2 matrices of signed 8-bit integers.

The core is an output-stationary systolic array: four multiply-accumulate cells
(PEs). Activations enter from the left and move one cell to the right every clock;
weights enter from the top and move down. Inputs are skewed so that every matching
pair meets in the right cell on the right clock. Each PE keeps its own result
C[i][j] in an 18-bit accumulator. A 2x2 multiply takes 4 compute clocks.

## Command interface

One command per clock while `VALID` (uio[7]) is high. `OP` is uio[2:0], data in on
ui[7:0], data out on uo[7:0].

| OP | Name      | Action |
|----|-----------|--------|
| 0  | NOP       | nothing |
| 1  | LOAD_A    | write DIN to the A buffer (row-major), pointer auto-increments |
| 2  | LOAD_B    | write DIN to the B buffer (row-major), pointer auto-increments |
| 3  | RUN       | start the multiply; BUSY (uio[5]) high, then DONE (uio[6]) high |
| 4  | READ      | DOUT <= next result byte (3 bytes per result, little-endian, two's complement) |
| 5  | CLEAR_PTR | reset pointers and DONE |
| 6  | SET_CFG   | bit 0 enables ReLU |
| 7  | READ_ID   | DOUT <= 0xE0 |

## How to test

1. Reset, then send READ_ID and check that DOUT reads 0xE0.
2. CLEAR_PTR, SET_CFG(0), four LOAD_A bytes, four LOAD_B bytes, RUN.
3. Wait for DONE, then send READ twelve times and assemble four 24-bit results.

Example: A = identity, B = [5, -7, 100, -128] returns C = B.

## External hardware

None. The RP2040 on the Tiny Tapeout demo board can drive all pins.
