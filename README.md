# quasar

A fast C++17 state-vector quantum circuit simulator for Apple Silicon, with a
gate-fusion pass and an Apple Accelerate (`cblas_zgemm`) dense-gate kernel path.
Every benchmark is verified to float epsilon against a Qiskit Aer golden state.

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

## Benchmarks (honest)

Measured on Apple Silicon (RelWithDebInfo). Golden baseline: **Qiskit Aer 2.2.3**
(`Statevector.from_instruction`); correctness gate is max-abs-diff ≤ 1e-9.

| Circuit   | dim     | max-abs-diff vs Aer | quasar (mean) | golden (Aer) |
|-----------|---------|---------------------|---------------|--------------|
| QFT-20    | 2²⁰     | 1.7e-18             | ~305 ms       | ~1330 ms     |
| GHZ-16    | 2¹⁶     | 1.1e-16             | ~1.2 ms       | ~5.8 ms      |
| QAOA-12   | 2¹²     | 1.1e-16             | ~0.9 ms       | ~7.3 ms      |
| Grover-10 | 2¹⁰     | 1.7e-12             | ~2.0 ms       | ~2670 ms*    |

\* The Grover golden time is dominated by Qiskit's multi-controlled-X
decomposition during `from_instruction`, not raw simulation — so it overstates
Aer's true sim cost. Treat it as "transpile + simulate," not a clean sim number.

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

(Golden Aer times: QFT-18 ~255 ms, QAOA-18 ~189 ms. Correctness gate still
max-abs-diff ≤ 1e-9 in every mode.)

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
