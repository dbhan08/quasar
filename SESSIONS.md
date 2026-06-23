# quasar — Sessions

Discrete 1–3 hour sessions. Each is independent enough to pause/resume. Check items off
as completed; append `Completed: YYYY-MM-DD` when all items in a session are done.

---

## Session 1 — Scaffold + state-vector core
- [x] CMake project builds; `quasar` CLI stub runs and prints version.
- [x] `StateVector` type: complex amplitude array, init to |0…0⟩, n-qubit register.
- [x] Single-qubit gate application by qubit index (the stride/pair sweep), with X, H, Z.
- [x] `probabilities()` and `sample()` (measurement in computational basis).
- [x] GoogleTest wired up; tests for H|0⟩ = |+⟩, X|0⟩ = |1⟩, normalization preserved.
- **Done when:** `ctest` passes and `quasar` can build + measure a 1-qubit Hadamard.
Completed: 2026-06-19

## Session 2 — Full gate set + two-qubit gates + circuit IR
- [x] Gate set: Y, S, T, phase, RX, RY, RZ (parametric).
- [x] Two-qubit gates: CNOT, CZ, SWAP; arbitrary single-control controlled gates.
- [x] `Circuit` IR: ordered list of (gate, target qubits, params) + a programmatic builder.
- [x] Apply a whole `Circuit` to a `StateVector`.
- [x] Tests: Bell state from H+CNOT, GHZ on 3 qubits, parametric-rotation identities.
- **Done when:** `ctest` passes; a Bell and a 3-qubit GHZ produce the exact known states.
Completed: 2026-06-19

## Session 3 — Reference algorithms + OpenQASM-subset import
- [x] Implement QFT(n), Grover (small oracle), GHZ(n), a QAOA ansatz as `Circuit`s.
- [x] OpenQASM 2.0-subset parser → `Circuit` (enough for the above + Aer interop).
- [x] Tests: QFT on a basis state matches the analytic DFT amplitudes; Grover amplifies
      the marked state above threshold.
- [x] **Create `README.md`** (living doc): pitch, build, run one circuit. Update each session.
- **Done when:** `ctest` passes; QFT and Grover verified against analytic expectations.
Completed: 2026-06-19

## Session 4 — Benchmark harness vs Qiskit Aer (correctness + latency)
- [x] Python script: build the same circuit in Qiskit, run Aer statevector, dump the
      golden state + Aer timing to JSON.
- [x] `quasar bench`: run the circuit, compare state to the Aer golden (max-abs-diff
      gate, ≤ 1e-9), print a latency table (quasar vs Aer, mean over N runs).
- [x] Reproducible: one command per circuit (QFT-20, Grover, QAOA), warmup + N iters.
- [x] Tests: bench harness self-check on a tiny circuit (states agree, table renders).
- **Done when:** `quasar bench` runs QFT-20 end-to-end, passes the correctness gate, and
      prints an honest quasar-vs-Aer latency table.
Completed: 2026-06-19 (baseline: qiskit-aer 2.2.3 via Statevector.from_instruction)

## Session 5 — Gate-fusion pass
- [x] IR pass: merge maximal runs of consecutive gates on overlapping qubit sets into a
      single fused multi-qubit unitary (cap fused width, e.g. ≤ 4 qubits).
- [x] Apply fused multi-qubit unitary to the state vector (general dense application).
- [x] Correctness: fused circuit state == unfused state (every reference circuit) and
      still passes the Aer gate.
- [x] Bench: report fused vs unfused latency on QFT / QAOA; record the speedup.
      NOTE: with the *scalar* gather/scatter apply path, fusion is currently SLOWER than
      the single-qubit sweeps (QFT-20: ~305 ms unfused vs ~2020 ms fused). This is the
      expected setup for Session 6: the dense fused blocks only pay off once routed
      through cblas_zgemm. Recorded honestly.
- **Done when:** `ctest` passes, fusion is state-for-state identical to unfused, and the
      bench shows the fused-vs-unfused delta.
Completed: 2026-06-19

## Session 6 — Accelerate dense-gate kernel path + profiler
- [x] Route dense multi-qubit gate application through `cblas_zgemm` (Accelerate,
      double-precision complex).
- [x] Per-stage profiler (`quasar profile`): times the scalar gather/scatter path vs
      the zgemm path per fused block width k.
- [x] Characterize: zgemm overtakes the scalar gather/scatter at fused block size
      **k=2 (dim 4)**, and the advantage grows: ~4x at k=4, ~15x at k=6, ~29x at k=8
      (N=22). This is the headline finding.
- [x] Tests: zgemm path produces bit-comparable states to the scalar path (~1e-12),
      on random gates and all reference circuits.
- **Done when:** `ctest` passes; profiler reports the sweep-vs-matrix crossover, and the
      bench reflects the kernel path on at least one algorithm.
      HONEST CAVEAT: at the per-kernel level zgemm wins from k=2 up. End-to-end on QFT-20,
      zgemm makes the *fused* path ~3.7x faster than fused-scalar (2011->551 ms), but the
      *unfused* single-qubit sweeps (305 ms) are still fastest for QFT because its fused
      blocks are dominated by diagonal controlled-phase gates where dense GEMM does
      redundant work. Fusion+zgemm is a net win on denser blocks, not on sparse ones.
Completed: 2026-06-19

## Session 6b — Diagonal-aware apply + threading
- [x] Diagonal-aware fusion: after composing a fused block's 2^k x 2^k unitary, detect
      if it is diagonal (all off-diagonal entries below 1e-12). If so, emit a compact
      `fused_diag` op carrying only the 2^k diagonal entries.
- [x] New `StateVector::applyDiagonal`: O(2^n) elementwise scaling by the diagonal entry
      selected by each amplitude's target-bit pattern — no gather/scatter, no GEMM. Routed
      regardless of the `--blas` flag.
- [x] Multi-threading (Apple GCD `dispatch_apply`): parallelized the single-qubit sweep
      (`apply1`), the dense `applyMulti` per-group loop, `applyDiagonal`, and the zgemm
      gather/scatter. Each worker writes a disjoint amplitude range/group set — race-free.
      Below a 2^14 work threshold the loops stay serial to avoid dispatch overhead.
- [x] Tests: (a) a phase/controlled-phase run produces a `fused_diag` block and a state
      identical (~1e-12) to the unfused apply, with and without `--blas`; (b) the threaded
      fused apply matches a trusted single-threaded reference on a 12-qubit mixed circuit.
- **Done when:** `ctest` passes (44/44); the Aer gate still passes (max-abs-diff <= 1e-9);
      the fused path beats the unfused path end-to-end on QFT-18 and QAOA-18.
      RESULT (18 qubits, mean ms): the fused path now WINS.
        QFT-18 : unfused 66.6->63.6 | fused 431.0->44.1 | fused+zgemm 128.9->55.7
        QAOA-18: unfused 66.9->46.3 | fused 188.6->29.2 | fused+zgemm 67.8->43.5
      Diagonal-aware fusion is the big lever (~10x on the fused path): QFT emits 39/60
      blocks as diag, QAOA 10/26. With diagonal blocks bypassing GEMM, the plain `--fuse`
      path now beats `--fuse --blas` — the remaining dense blocks are small (k<=4) and the
      threaded scalar apply beats the GEMM gather/scatter overhead at that width. Threading
      gave a modest sublinear speedup on the sweep-bound paths (unfused QFT 66.6->63.6,
      ~1.05x; unfused QAOA 66.9->46.3, ~1.44x) — memory-bandwidth-bound, not core-bound, so
      well short of 4-8x. The diagonal win dominates.
Completed: 2026-06-19

## Session 7 — Polish + docs + demo
- [x] Finalize `README.md`: pitch, "run it", usage examples, "how it works" (2–4 bullets),
      stack list, and an honest bench table with the crossover finding.
- [x] Write `DEMO.md`: reproducible command sequence (build → run QFT → bench vs Aer →
      show fusion speedup) with expected outputs. Document only what actually works.
- [x] Clean CLI help, error messages, and `--help` for every subcommand.
- **Done when:** a fresh clone can follow `DEMO.md` to reproduce a correct run + the bench.
Completed: 2026-06-19

## Session 8 — QAOA portfolio-optimization capstone
- [x] `yfinance` fetcher: pull historical daily returns for ~8 chosen tickers; compute
      expected-return vector μ and covariance Σ. (try/except + `--offline`; live fetch
      refreshes the bundled `capstone/data/returns_sample.csv` for reproducibility.)
- [x] Encode "select K of N assets, maximize μ-tilt − risk penalty" as an Ising/QUBO →
      a cost Hamiltonian. (QUBO with cardinality penalty → Ising Z fields h_i + ZZ J_ij.)
- [x] Build the QAOA circuit (cost + mixer layers, p≥1); run it on the simulator; sample
      the bitstring → chosen basket. (Emitted as OpenQASM 2.0 — only h/rz/cx/rx, all
      already parsed by quasar — bridged via `./build/quasar run` for full probabilities.)
- [x] Classical check: brute-force the optimum over all baskets (feasible at N=8);
      confirm QAOA converges to / near the optimum.
- [x] Tests: tiny portfolio (N=4) where the optimal basket is hand-verifiable; QAOA
      objective within tolerance of brute-force; QUBO↔Ising round-trip. (5 pytest cases,
      all green via `.venv/bin/python -m pytest capstone/`.)
- [x] Update `README.md` with the capstone demo + the honesty framing (optimization, not
      prediction; no edge, no money).
- **Done when:** `quasar` runs QAOA on a real ~8-stock portfolio and its chosen basket
      matches the brute-force optimum within tolerance, with tests passing.
      RESULT: live yfinance pull of 8 tickers × 500 trading days (~2y), K=4. p=2 QAOA on
      quasar selects [GOOGL, JPM, XOM, JNJ] = the brute-force optimum (cost gap 0.0).
      No C++ changes needed — `quasar run` already prints the full probability vector;
      `ctest` stays 44/44. HONEST CAVEAT: p=1 does NOT reach the optimum on this instance
      (p=2 does), and the result is sensitive to the cardinality-penalty scale — a
      too-large penalty flattens the QAOA distribution to near-uniform and the optimum is
      lost, so a just-binding penalty is used.
Completed: 2026-06-23

## Session 9 — Live solver-comparison dashboard
- [ ] Streamlit dashboard that, on a refresh timer, re-pulls recent prices and re-solves
      the portfolio three ways: **QAOA-on-quasar**, **classical optimum**, **equal-weight**.
- [ ] Comparison panel: each solver's chosen basket, objective value, QAOA-vs-optimum
      quality gap, and solver runtime — updating as data refreshes.
- [ ] Illustrative forward track: paper-money value of each basket over time, clearly
      labeled "paper money, illustrative — not a trading signal, not an edge."
- [ ] Guard the framing in-UI: a visible disclaimer that this compares *solver behavior*,
      and that baskets are expected to nearly coincide at small N.
- [ ] Smoke test: dashboard renders, all three solvers return a basket, no crash on a
      data-refresh cycle.
- **Done when:** the dashboard runs locally, refreshes, and shows the three solvers side
      by side with the honesty disclaimer visible.

## Session 10 — Final docs + ship
- [ ] Finalize `README.md` + `DEMO.md`: core sim demo, the QAOA capstone, and the
      dashboard (screenshot or shot list), with honest bench numbers.
- [ ] Verify `.gitignore` excludes `RESUME.md`, `.env`, build artifacts.
- [ ] Commit per logical unit; push. (See Ship flow.)
- **Done when:** repo is public/pushed, green, README pitch as the description.

---

> v2 ideas (out of scope now): density-matrix + noise model; tensor-network contraction
> mode for low-entanglement circuits; multi-threaded gate application; larger-qubit
> out-of-core simulation; deeper QAOA (higher p) + warm-start; more assets via problem
> decomposition.
