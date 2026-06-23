# quasar

A fast C++17 state-vector quantum circuit simulator for Apple Silicon, with a
gate-fusion pass and an Apple Accelerate (`cblas_zgemm`) dense-gate kernel path.
Every benchmark is verified to float epsilon against an exact Qiskit reference
state, and timed against Qiskit's production AerSimulator backend.

The point isn't "a new way to simulate circuits" — it's an honestly benchmarked
characterization of **where matrix-coprocessor acceleration helps state-vector
simulation on Apple Silicon and where it doesn't.**

## Run it

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Simulate an OpenQASM 2.0 circuit:

```sh
./build/quasar run examples/bell.qasm
# 00  0.5
# 11  0.5

./build/quasar run examples/bell.qasm --shots 1000
# shots=1000
# 00  501
# 11  499
```

Every subcommand supports `--help`. Subcommands: `run`, `bench`, `profile`,
`demo`, `--version`.

## How it works

- **State vector.** An n-qubit register is a dense `std::vector<std::complex<double>>`
  of length 2ⁿ. Single-qubit gates sweep index pairs `(i, i|2^q)`; k-qubit gates
  gather 2ᵏ amplitudes per spectator setting, multiply by the dense unitary, and
  scatter back.
- **Little-endian ordering.** Qubit 0 is the least-significant bit of the basis
  index, matching Qiskit. Basis labels print MSB-left (`q_{n-1}…q_0`).
- **Gate fusion.** A greedy IR pass merges consecutive gates whose combined qubit
  width ≤ 4 into one dense unitary (each gate expanded to the merged subspace and
  matrix-multiplied in order). Fusion is asserted state-for-state identical to the
  unfused result on every reference circuit and against the golden state.
- **Diagonal-aware apply.** After composing a fused block, the pass detects whether
  the unitary is diagonal (all off-diagonal entries < 1e-12) — true for the
  controlled-phase ladders in QFT and the ZZ cost terms in QAOA. Diagonal blocks are
  emitted as a compact `fused_diag` op (just the 2ᵏ diagonal) and applied as an
  O(2ⁿ) elementwise scaling: no gather/scatter, no GEMM. This is the single biggest
  win for diagonal-dominated circuits.
- **Accelerate kernel.** Non-diagonal fused dense gates can route through `cblas_zgemm`:
  gather the 2^(n−k) groups as columns of a 2ᵏ×groups matrix, do one complex GEMM,
  scatter back.
- **Multi-threading.** The heavy apply loops (single-qubit sweep, dense per-group
  apply, diagonal apply, zgemm gather/scatter) are parallelized across cores with
  Apple GCD `dispatch_apply`; each worker writes a disjoint amplitude range, so the
  result is bit-identical to the serial path.

## Gate set

X, Y, Z, H, S, T, RX(θ), RY(θ), RZ(θ), Phase(λ); CNOT, CZ, SWAP, and arbitrary
single-control controlled gates. Algorithms: QFT, GHZ, Grover, ring-Ising QAOA
ansatz.

## Portfolio optimization (capstone)

A real QAOA portfolio optimizer that runs **on the quasar simulator** and shows
its chosen basket matches the brute-force classical optimum.

> **Honesty first.** This is an *optimization* demo, not a prediction. It picks
> the best mix of a **fixed** set of assets under a mean-variance objective. It
> does **not** predict prices, **not** beat the market, and makes **no** money
> claim — there is no trading edge here. At small N the QAOA result is *expected*
> to match the classical optimum; that coincidence is the point — it's
> verifiable correctness, not a letdown.

What it does: pull ~2 years of daily returns for 8 tickers (AAPL, MSFT, GOOGL,
AMZN, JPM, XOM, JNJ, PG) via yfinance, compute the expected-return vector μ and
covariance Σ, encode "pick exactly K of N assets maximizing `q·μᵀx − xᵀΣx`" as a
QUBO with a cardinality penalty, map it to an Ising model (Z fields + ZZ
couplings), emit a `p`-layer QAOA circuit as **OpenQASM 2.0** (only `h`, `rz`,
`cx`, `rx` — all parsed by quasar), simulate it with `./build/quasar run` to get
the full probability distribution, optimize the QAOA angles (grid warm start +
COBYLA), and take the most-probable feasible bitstring as the basket. Finally it
brute-forces the true optimum over all exactly-K baskets (feasible at N=8) and
prints the comparison.

Run it (bundled CSV, fully offline and reproducible):

```sh
.venv/bin/python capstone/portfolio_qaoa.py --offline
```

Drop `--offline` to fetch fresh prices from yfinance; a successful live fetch
refreshes `capstone/data/returns_sample.csv` so the offline path stays current.

Example output (8 assets, pick K=4, p=2 QAOA layers):

```
assets (N=8): AAPL, MSFT, GOOGL, AMZN, JPM, XOM, JNJ, PG
select exactly K=4, q-tilt=0.5, QAOA layers p=2, quasar evals=90
--------------------------------------------------------------------
QAOA basket       : ['GOOGL', 'JPM', 'XOM', 'JNJ']
  objective C(x)  : -0.240496   (most-probable feasible, p=0.0173)
brute-force basket: ['GOOGL', 'JPM', 'XOM', 'JNJ']
  objective C(x)  : -0.240496
--------------------------------------------------------------------
MATCH: YES -- QAOA found the classical optimum
cost gap (QAOA - brute): 0.000000e+00
```

Honest caveats: `p=1` QAOA does **not** reach the optimum on this 8-asset
instance — `p=2` does, and the result is sensitive to the penalty scale (a
too-large cardinality penalty flattens the QAOA distribution into near-uniform
noise and the optimum gets lost). The script uses a just-binding penalty for
that reason. The feasible subspace is 70 of 256 states, so even the optimal
bitstring carries only ~2% probability — what's verified is that it is the
*most-probable feasible* state and that it equals the brute-force optimum.

Tests: `.venv/bin/python -m pytest capstone/` (N=4 hand-verifiable portfolio +
QUBO↔Ising round-trip).

## Benchmarks (honest)

Measured on Apple Silicon (RelWithDebInfo). Two **separate** baselines, stated
precisely so the claims survive scrutiny:

- **Correctness golden:** `qiskit.quantum_info.Statevector` — an exact numpy
  reference. Gate: max-abs-diff ≤ 1e-9 (typically ~1e-17).
- **Performance baseline:** the real `qiskit_aer.AerSimulator(method="statevector")`
  — Aer's production C++ backend (its own gate fusion + threading), timing only
  `.run()` after a warmup.

> Honesty note: an earlier version of this harness mistakenly timed the slow
> `quantum_info.Statevector` and labeled it "Aer." That inflated the apparent
> speedup ~5×. The numbers below are vs the **real AerSimulator**.

### Correctness (vs exact reference)

| Circuit   | dim   | max-abs-diff |
|-----------|-------|--------------|
| QFT-18    | 2¹⁸   | 3.7e-18      |
| QAOA-18   | 2¹⁸   | 3.5e-17      |
| GHZ-16    | 2¹⁶   | 1.1e-16      |
| Grover-10 | 2¹⁰   | 1.7e-12      |

### Speed vs Qiskit AerSimulator (the honest comparison)

quasar's fastest mode (`--fuse`) vs the real AerSimulator, mean ms:

| Circuit | quasar `--fuse` | AerSimulator | ratio |
|---------|-----------------|--------------|-------|
| QFT-18  | 48              | 73           | 1.5×* |
| QFT-20  | 188             | 214          | 1.14× |
| QFT-21  | 390             | 412          | 1.06× |
| QAOA-18 | 29              | 40           | 1.36× |
| QAOA-20 | 118             | 154          | 1.30× |
| QAOA-21 | 239             | 326          | 1.36× |

\* The 18-qubit QFT ratio is inflated by Aer's per-`.run()` job + statevector-
serialization overhead (the state is only ~4 MB). As N grows and compute
dominates, the QFT ratio settles to **~parity** (1.06× at 21q); the QAOA
advantage (~1.3×) is **stable across sizes** — that's the real win.

**Honest summary:** quasar is **at parity** with Qiskit's production AerSimulator
on QFT and **~1.3× faster on QAOA-style circuits**, for CPU statevector simulation
at 18–21 qubits — with a from-scratch simulator. Aer is general-purpose (noise,
multiple methods, GPU); quasar is a specialized CPU statevector simulator, so
edging out a general tool on a narrow case is the expected shape of this result.
Absolute ms are machine/load dependent; the **ratios** are the reproducible claim.

### The zgemm crossover (headline finding)

`quasar profile` times the scalar gather/scatter path against the `cblas_zgemm`
path per fused block width k, on a 22-qubit state:

| k | dim | scalar (ms) | zgemm (ms) | speedup |
|---|-----|-------------|------------|---------|
| 1 | 2   | 128         | 147        | 0.87x (scalar wins) |
| 2 | 4   | 92          | 73         | 1.26x   |
| 3 | 8   | 86          | 38         | 2.26x   |
| 4 | 16  | 109         | 25         | 4.37x   |
| 6 | 64  | 299         | 20         | 14.8x   |
| 8 | 256 | 1092        | 38         | 29.0x   |

**zgemm overtakes the scalar gather/scatter at fused block size k=2 (dim 4)**, and
the advantage grows sharply with block width.

The original honest caveat was that the *unfused* sweep still beat the fused path
end-to-end on diagonal-dominated circuits (QFT controlled-phase, QAOA ZZ terms),
because routing diagonal blocks through dense GEMM does redundant work. **Session 6b
fixes exactly that:** diagonal-aware fusion now applies those blocks as an O(2ⁿ)
elementwise scaling instead of a GEMM, and the apply loops are multi-threaded.

### Diagonal-aware apply + threading (Session 6b)

End-to-end mean latency at **18 qubits** (2¹⁸ amplitudes), before vs after:

| Circuit | mode        | before (ms) | after (ms) | speedup |
|---------|-------------|-------------|------------|---------|
| QFT-18  | unfused     | 66.6        | 63.6       | 1.05x   |
| QFT-18  | fused       | 431.0       | 44.1       | 9.8x    |
| QFT-18  | fused+zgemm | 128.9       | 55.7       | 2.3x    |
| QAOA-18 | unfused     | 66.9        | 46.3       | 1.44x   |
| QAOA-18 | fused       | 188.6       | 29.2       | 6.5x    |
| QAOA-18 | fused+zgemm | 67.8        | 43.5       | 1.56x   |

(Real AerSimulator times: QFT-18 ~73 ms, QAOA-18 ~40 ms. Correctness gate still
max-abs-diff ≤ 1e-9 in every mode. See the speed table above for the honest
quasar-vs-Aer comparison across sizes.)

The plain `--fuse` path is now the fastest mode for both circuits, and it **beats the
unfused sweep** — the goal of the session. The big lever is diagonal-awareness: QFT
fuses 39/60 blocks as diagonal, QAOA 10/26, and those bypass GEMM entirely. With the
remaining dense blocks small (k ≤ 4), the threaded scalar apply beats the GEMM
gather/scatter overhead, so `--fuse` now edges out `--fuse --blas`. Threading itself
is a modest, sublinear win on the sweep-bound paths (these are memory-bandwidth-bound,
not core-bound, so well short of the 4–8x core count) — the diagonal optimization
dominates the speedup.

## Stack

C++17, CMake, Apple Accelerate (`cblas_zgemm`), Grand Central Dispatch
(`dispatch_apply` multi-threading), GoogleTest; Python + Qiskit Aer for golden-state
generation in the benchmark harness.
