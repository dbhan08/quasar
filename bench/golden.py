#!/usr/bin/env python3
"""Golden-state + timing generator for the quasar benchmark harness.

Builds a circuit as quasar-IR (a list of gate ops identical to what quasar's C++
`algorithms` emit), applies the SAME IR to a Qiskit circuit, runs Aer's
statevector simulator as the golden reference, and dumps:

  - <name>.circuit.json : the quasar IR (nqubits + ops) for quasar to load
  - <name>.golden.json  : the golden statevector (re/im per amplitude) + Aer
                          timing, in little-endian basis ordering

Qiskit is little-endian (qubit 0 = LSB), matching quasar, so the basis index
ordering lines up directly.

Baseline: Qiskit Aer statevector (this script requires qiskit + qiskit-aer).
"""

import argparse
import json
import math
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# ---- quasar-IR builders (mirror src/algorithms.cpp exactly) ----

def ir_qft(n):
    ops = []
    for j in range(n - 1, -1, -1):
        ops.append({"name": "h", "qubits": [j], "params": []})
        for k in range(j - 1, -1, -1):
            angle = math.pi / (2.0 ** (j - k))
            ops.append({"name": "cphase", "qubits": [k, j], "params": [angle]})
    for i in range(n // 2):
        ops.append({"name": "swap", "qubits": [i, n - 1 - i], "params": []})
    return {"nqubits": n, "ops": ops}


def ir_ghz(n):
    ops = [{"name": "h", "qubits": [0], "params": []}]
    for i in range(n - 1):
        ops.append({"name": "cx", "qubits": [i, i + 1], "params": []})
    return {"nqubits": n, "ops": ops}


def ir_grover(n, marked):
    ops = []
    N = 1 << n
    iters = int(math.floor(math.pi / 4.0 * math.sqrt(N)))

    def hadamards():
        for q in range(n):
            ops.append({"name": "h", "qubits": [q], "params": []})

    def phase_flip(target):
        for q in range(n):
            if not ((target >> q) & 1):
                ops.append({"name": "x", "qubits": [q], "params": []})
        ops.append({"name": "mcz", "qubits": list(range(n)), "params": []})
        for q in range(n):
            if not ((target >> q) & 1):
                ops.append({"name": "x", "qubits": [q], "params": []})

    hadamards()
    for _ in range(iters):
        phase_flip(marked)
        hadamards()
        phase_flip(0)
        hadamards()
    return {"nqubits": n, "ops": ops}


def ir_qaoa(n, gammas, betas):
    ops = [{"name": "h", "qubits": [q], "params": []} for q in range(n)]
    p = min(len(gammas), len(betas))
    for layer in range(p):
        g, b = gammas[layer], betas[layer]
        edges = (1 if n == 2 else (0 if n < 2 else n))
        for e in range(edges):
            i, j = e, (e + 1) % n
            ops.append({"name": "cx", "qubits": [i, j], "params": []})
            ops.append({"name": "rz", "qubits": [j], "params": [2.0 * g]})
            ops.append({"name": "cx", "qubits": [i, j], "params": []})
        for q in range(n):
            ops.append({"name": "rx", "qubits": [q], "params": [2.0 * b]})
    return {"nqubits": n, "ops": ops}


# ---- apply quasar-IR to a Qiskit circuit ----

def to_qiskit(ir):
    n = ir["nqubits"]
    qc = QuantumCircuit(n)
    for op in ir["ops"]:
        name = op["name"]
        q = op["qubits"]
        p = op["params"]
        if name == "h":
            qc.h(q[0])
        elif name == "x":
            qc.x(q[0])
        elif name == "y":
            qc.y(q[0])
        elif name == "z":
            qc.z(q[0])
        elif name == "s":
            qc.s(q[0])
        elif name == "t":
            qc.t(q[0])
        elif name == "rx":
            qc.rx(p[0], q[0])
        elif name == "ry":
            qc.ry(p[0], q[0])
        elif name == "rz":
            qc.rz(p[0], q[0])
        elif name == "phase":
            qc.p(p[0], q[0])
        elif name == "cx":
            qc.cx(q[0], q[1])
        elif name == "cz":
            qc.cz(q[0], q[1])
        elif name == "swap":
            qc.swap(q[0], q[1])
        elif name == "cphase":
            qc.cp(p[0], q[0], q[1])
        elif name == "crz":
            qc.crz(p[0], q[0], q[1])
        elif name == "mcz":
            # Multi-controlled Z on all listed qubits: controls q[:-1], target
            # last. Z = H X H, so MCZ = H(target) MCX H(target).
            ctrls = q[:-1]
            tgt = q[-1]
            qc.h(tgt)
            qc.mcx(ctrls, tgt)
            qc.h(tgt)
        else:
            raise ValueError(f"unsupported gate in golden harness: {name}")
    return qc


def golden_state(qc, reps=5):
    # Aer statevector via quantum_info is exact; time the evolution.
    best = None
    sv = None
    for _ in range(reps):
        t0 = time.perf_counter()
        sv = Statevector.from_instruction(qc)
        t1 = time.perf_counter()
        dt = t1 - t0
        if best is None or dt < best:
            best = dt
    return np.asarray(sv.data), best


def build_ir(name, n):
    if name == "qft":
        return ir_qft(n)
    if name == "ghz":
        return ir_ghz(n)
    if name == "grover":
        # Mark the all-ones state by default.
        return ir_grover(n, (1 << n) - 1)
    if name == "qaoa":
        gammas = [0.4 * (k + 1) for k in range(2)]
        betas = [0.3 / (k + 1) for k in range(2)]
        return ir_qaoa(n, gammas, betas)
    raise ValueError(f"unknown circuit: {name}")


def main():
    ap = argparse.ArgumentParser(description="quasar golden-state generator")
    ap.add_argument("--circuit", required=True,
                    choices=["qft", "ghz", "grover", "qaoa"])
    ap.add_argument("--qubits", type=int, required=True)
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    ir = build_ir(args.circuit, args.qubits)
    qc = to_qiskit(ir)
    data, aer_time = golden_state(qc, args.reps)

    with open(args.out + ".circuit.json", "w") as f:
        json.dump(ir, f)

    golden = {
        "nqubits": args.qubits,
        "circuit": args.circuit,
        "baseline": "qiskit-aer",
        "aer_time_s": aer_time,
        "re": [float(z.real) for z in data],
        "im": [float(z.imag) for z in data],
    }
    with open(args.out + ".golden.json", "w") as f:
        json.dump(golden, f)

    print(f"wrote {args.out}.circuit.json and {args.out}.golden.json "
          f"(aer_time={aer_time*1e3:.3f} ms, dim={len(data)})")


if __name__ == "__main__":
    main()
