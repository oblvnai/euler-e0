# SPDX-License-Identifier: Apache-2.0
# Planck Silicon — Euler E0 cocotb tests (run by Tiny Tapeout's GitHub Actions,
# both on the RTL and on the final gate-level netlist).

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge

N = 2  # array size of the taped-out configuration

OP_NOP, OP_LOAD_A, OP_LOAD_B, OP_RUN, OP_READ, OP_CLEAR, OP_CFG, OP_ID = range(8)


async def cmd(dut, op, data=0):
    """Issue one command: valid=1 for exactly one clock."""
    await FallingEdge(dut.clk)
    dut.uio_in.value = 0x80 | op
    dut.ui_in.value = data & 0xFF
    await FallingEdge(dut.clk)
    dut.uio_in.value = 0
    dut.ui_in.value = 0


def golden(a, b, relu):
    c = []
    for i in range(N):
        for j in range(N):
            s = sum(a[i * N + k] * b[k * N + j] for k in range(N))
            c.append(max(s, 0) if relu else s)
    return c


async def matmul(dut, a, b, relu):
    await cmd(dut, OP_CLEAR)
    await cmd(dut, OP_CFG, 1 if relu else 0)
    for v in a:
        await cmd(dut, OP_LOAD_A, v)
    for v in b:
        await cmd(dut, OP_LOAD_B, v)
    await cmd(dut, OP_RUN)
    for _ in range(20):
        if int(dut.uio_out.value) & 0x40:  # DONE
            break
        await ClockCycles(dut.clk, 1)
    assert int(dut.uio_out.value) & 0x40, "DONE never asserted"
    out = []
    for _ in range(N * N):
        raw = 0
        for byte in range(3):
            await cmd(dut, OP_READ)
            raw |= int(dut.uo_out.value) << (8 * byte)
        out.append(raw - (1 << 24) if raw & 0x800000 else raw)
    return out


async def reset(dut):
    clock = Clock(dut.clk, 20, unit="ns")  # 50 MHz
    cocotb.start_soon(clock.start())
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


@cocotb.test()
async def test_chip_id(dut):
    await reset(dut)
    await cmd(dut, OP_ID)
    assert int(dut.uo_out.value) == 0xE0
    assert int(dut.uio_oe.value) == 0b0110_0000


@cocotb.test()
async def test_matmul(dut):
    await reset(dut)
    rng = random.Random(20261122)
    cases = [
        ([1, 0, 0, 1], [5, -7, 100, -128], False),   # identity
        ([-128] * 4, [-128] * 4, False),              # largest positive result
        ([-128] * 4, [127] * 4, False),               # most negative result
        ([-128] * 4, [127] * 4, True),                # ReLU clamps to zero
    ]
    for t in range(60):
        a = [rng.randint(-128, 127) for _ in range(N * N)]
        b = [rng.randint(-128, 127) for _ in range(N * N)]
        cases.append((a, b, bool(t & 1)))
    for a, b, relu in cases:
        got = await matmul(dut, a, b, relu)
        exp = golden(a, b, relu)
        assert got == exp, f"A={a} B={b} relu={relu}: got {got}, expected {exp}"
    dut._log.info(f"{len(cases)} matrix multiplies verified")
