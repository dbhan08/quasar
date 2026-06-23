# quasar — Demo

A reproducible command sequence from a fresh clone. Every output below was
actually measured on Apple Silicon; your latency numbers will vary, but the
correctness numbers and the crossover should reproduce.

## 0. Build + test

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Expected tail:

```
100% tests passed, 0 tests failed out of 42
```

## 1. Run a circuit from OpenQASM

```sh
./build/quasar run examples/bell.qasm
```
```
00  0.5
11  0.5
```

Sample it (seeded, so counts are reproducible):

```sh
./build/quasar run examples/bell.qasm --shots 1000
```
```
shots=1000
00  501
11  499
```

A 3-qubit GHZ:

```sh
./build/quasar run examples/ghz3.qasm
```
```
000  0.5
111  0.5
```

## 2. Bench vs the Qiskit Aer golden

First generate the golden state + Aer timing (needs the venv with qiskit-aer):

```sh
python3 -m venv .venv
.venv/bin/pip install --quiet qiskit qiskit-aer numpy
.venv/bin/python bench/golden.py --circuit qft --qubits 20 --out bench_out/qft_20
```
```
wrote bench_out/qft_20.circuit.json and bench_out/qft_20.golden.json (aer_time=... ms, dim=1048576)
```

Then run quasar against it:

```sh
./build/quasar bench --circuit qft --qubits 20 -n 5
```
```
circuit=qft qubits=20 dim=1048576 baseline=qiskit-aer mode=unfused
correctness: max-abs-diff=1.73472e-18  PASS (<=1e-9)
latency (mean over 5 runs):
  quasar : mean=305 ms  best=302 ms
  golden : 1330 ms (qiskit-aer)
```

quasar matches the Aer statevector to ~1e-18 and runs QFT-20 several times faster
than `Statevector.from_instruction`.

## 3. Gate fusion (same state, different kernel path)

```sh
./build/quasar bench --circuit qft --qubits 20 -n 5 --fuse   # scalar dense apply
./build/quasar bench --circuit qft --qubits 20 -n 5 --blas   # fused + cblas_zgemm
```

Both still print `correctness: ... PASS (<=1e-9)` — fusion and the zgemm path are
state-for-state identical to the unfused run. On QFT specifically, fused-scalar is
~2010 ms, fused+zgemm ~550 ms, and plain unfused ~305 ms (see README for why
unfused wins on this particular circuit).

## 4. The zgemm crossover (headline)

```sh
./build/quasar profile --qubits 22 --reps 7 --maxk 8
```
```
profiler: N=22 qubits, reps=7 (one dense k-qubit gate over all 2^(N-k) groups)
  k   dim    scalar(ms)   zgemm(ms)   speedup   winner
  1   2      128          147         0.87x     scalar
  2   4      92           73          1.26x     zgemm
  3   8      86           38          2.26x     zgemm
  4   16     109          25          4.37x     zgemm
  ...
crossover: zgemm overtakes the scalar path at fused block size k=2 (dim 4)
```

The `cblas_zgemm` path beats the hand-written gather/scatter loop from fused block
width **k=2** upward, with the gap widening as blocks get denser.
