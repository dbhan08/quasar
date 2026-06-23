# quasar

A fast state-vector quantum circuit simulator for Apple Silicon.

## Pitch

`quasar` simulates quantum circuits by evolving the full 2ⁿ complex state vector on
Apple Silicon. It runs standard quantum algorithms (QFT, Grover, GHZ, QAOA ansätze)
and gets its speed from two HPC techniques: **gate fusion** (collapsing runs of
consecutive gates into a single multi-qubit unitary) and routing the resulting dense
gate applications through **Apple Accelerate's matrix path** (the M-series matrix
coprocessor). Correctness and speed are both verified against Qiskit Aer on every
benchmark.

The contribution isn't "a new way to simulate quantum circuits" — it's a careful,
honestly-benchmarked characterization of *where matrix-coprocessor acceleration helps
state-vector simulation on Apple Silicon and where memory bandwidth caps it*.

## Goals

- Correct state-vector simulation for an n-qubit register (target: 25 qubits on 16 GB).
- A clean gate set: X, Y, Z, H, S, T, RX, RY, RZ, phase, plus CNOT, CZ, SWAP, and
  arbitrary controlled gates.
- A small circuit IR + a programmatic builder and OpenQASM 2.0-subset import.
- A **gate-fusion pass** that merges consecutive gates on overlapping qubits into one
  unitary applied as a single matrix operation.
- A dense-gate kernel path through Apple Accelerate (`cblas_cgemm`), with a profiler
  that reports per-stage time and where the matrix path engages vs. falls back.
- A reproducible benchmark harness that runs the *same* circuit through `quasar` and
  Qiskit Aer, gates correctness to float epsilon, and reports the latency delta.
- A **QAOA portfolio-optimization capstone**: encode "pick the best K of N stocks"
  (Markowitz risk/return) as an Ising/QUBO, build the QAOA circuit, run it on `quasar`,
  and compare the chosen basket against the brute-force classical optimum.
- A **live comparison dashboard** that, as market data refreshes, shows three solvers
  side by side — QAOA-on-`quasar`, the classical optimum, and an equal-weight baseline —
  comparing their chosen baskets, objective values, solution-quality gap, and solver
  runtime, plus an illustrative paper-money forward track of each basket.

## Non-goals

- No real quantum hardware / cloud QPU access. Pure classical simulation.
- No density-matrix / noise model in v1 (possible v2). State vector only.
- No tensor-network contraction mode in v1 (possible v2).
- Not trying to beat GPU simulators (qsim/cuStateVec) in absolute terms — the lane is
  the CPU matrix-coprocessor path on Apple Silicon, benchmarked honestly.
- The QAOA capstone **optimizes** a portfolio (best mix of given assets); it does **not**
  predict prices, beat the market, or generate a trading edge. The dashboard compares
  *solver behavior*, not profit. The paper-money forward track is illustrative only —
  no real money, no claim of alpha. Expect QAOA and classical to pick near-identical
  baskets at small N; the point is the quantum algorithm working, not outperforming.

## Stack

- **C++17** core, **CMake** build.
- **Apple Accelerate** (`cblas_cgemm` / vDSP) for dense gate kernels.
- **GoogleTest** for unit + correctness tests.
- **Python** (thin layer) for circuit construction sugar and the **Qiskit Aer**
  baseline + golden-state generation in the bench harness.
- **Python dashboard** (Streamlit) + **yfinance** (free historical/recent prices) for
  the QAOA portfolio capstone and the live solver-comparison dashboard.

## Scope

Multi-week. Core (~1 week): state vector + full gate set + reference algorithms +
gate-fusion pass + Accelerate dense-kernel path + correctness/latency bench vs Qiskit
Aer. Capstone (+~1 week): QAOA portfolio-optimization demo and the live
solver-comparison dashboard. Weekend MVP within the core = state vector +
single/two-qubit gates + QFT + one matrix path + a basic bench.

## Risks / unknowns

- **Complex GEMM payoff is uncertain.** Single- and two-qubit gates are cheap,
  bandwidth-bound vector sweeps; the matrix path only pays off once fusion produces
  large enough multi-qubit blocks. The honest result may be "fusion helps, the matrix
  coprocessor helps only past block size K" — and that finding *is* the deliverable.
- **Memory wall.** State vector is 16 bytes × 2ⁿ. 25 qubits ≈ 512 MB; 28 ≈ 4 GB. The
  simulator is memory-bandwidth-bound at scale, so kernel cleverness has a ceiling —
  worth measuring and stating, not hiding.
- **Qiskit Aer is a strong baseline** (vectorized, fused, multi-threaded). Matching it
  on CPU is the realistic bar; beating it everywhere is not expected. Honest framing
  required.
- **Gate fusion correctness.** Merging unitaries in the right order, with correct qubit
  index mapping, is the bug-prone part. Every fused circuit must be checked
  state-for-state against the unfused result and against Aer.
- **QAOA ≈ classical at small N.** With ~8 assets the classical optimum is brute-forceable
  and QAOA will (correctly) match it — so the capstone's value is "the quantum algorithm
  runs and converges to the optimum on my simulator," not superiority. Stated honestly,
  this is a feature (verifiable correctness), not a letdown.
- **"Real-time" is refresh-on-interval, not tick streaming.** The dashboard re-pulls
  prices and re-solves on a timer; portfolio optimization isn't a per-tick problem. The
  forward-performance track is illustrative paper money, not a backtest of an edge.

## Target resume bullet

> Built a C++17 state-vector quantum circuit simulator for Apple Silicon with a
> gate-fusion graph pass and matrix-coprocessor-accelerated dense-gate kernels;
> verified correctness to float epsilon and benchmarked latency against Qiskit Aer
> on QFT / Grover / QAOA circuits up to [measure: max qubits that fit in RAM].
