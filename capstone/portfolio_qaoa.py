#!/usr/bin/env python3
"""QAOA portfolio optimization on the quasar state-vector simulator.

HONESTY DISCLAIMER
==================
This is an OPTIMIZATION demo, not a prediction. It picks the best mix of a FIXED
set of assets under a mean-variance objective. It does NOT predict prices, does
NOT beat the market, and makes NO money claim. There is no trading edge here.
At small N the QAOA result is expected to match the brute-force classical
optimum -- that coincidence is the point: it is verifiable correctness, not a
letdown.

What it actually does
---------------------
1. DATA   : fetch ~8 tickers' daily returns (yfinance, ~2y), compute the
            expected-return vector mu and covariance Sigma. Falls back to a
            bundled CSV when offline / fetch fails.
2. ENCODE : "pick exactly K of N assets, maximize q*mu^T x - x^T Sigma x" as a
            QUBO with a cardinality penalty P*(sum x_i - K)^2; map x->z spins to
            get an Ising model (Z fields h_i, ZZ couplings J_ij).
3. CIRCUIT: emit an OpenQASM 2.0 QAOA circuit using only gates quasar parses
            (h, rz, cx, rx).
4. SIMULATE: run ./build/quasar to get the full probability distribution.
5. OPTIMIZE: tune (gammas, betas) with COBYLA / grid search; each evaluation
            emits QASM -> runs quasar -> computes E = sum_x P(x) C(x).
6. RESULT : take the most-probable FEASIBLE bitstring as the QAOA basket and
            compare it to the brute-force optimum over exactly-K baskets.
"""

import argparse
import itertools
import math
import os
import subprocess
import tempfile

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
QUASAR_BIN = os.path.join(REPO, "build", "quasar")
SAMPLE_CSV = os.path.join(HERE, "data", "returns_sample.csv")

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "XOM", "JNJ", "PG"]

DISCLAIMER = (
    "Optimization demo, not prediction. No trading edge, no money claim.\n"
    "At small N, QAOA is expected to match the classical optimum -- that is "
    "verifiable correctness, not a letdown."
)


# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------
def fetch_returns(tickers, period="2y", offline=False):
    """Return (tickers, daily-returns DataFrame). Live fetch refreshes the CSV;
    on any failure or --offline we load the bundled CSV instead.

    Returns (tickers, returns_df, source) where source is "live" or "csv".
    """
    if not offline:
        try:
            import pandas as pd
            import yfinance as yf

            raw = yf.download(
                tickers,
                period=period,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            # yfinance returns a column MultiIndex; grab adjusted close.
            close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
            close = close[tickers].dropna()
            rets = close.pct_change().dropna()
            if rets.shape[0] < 50:
                raise RuntimeError("too few rows returned from yfinance")
            # Refresh the bundled CSV so the demo stays reproducible offline.
            rets.to_csv(SAMPLE_CSV)
            return tickers, rets, "live"
        except Exception as e:  # noqa: BLE001 - network is intentionally flaky
            print(f"[data] live fetch failed ({e!r}); falling back to bundled CSV")

    import pandas as pd

    if not os.path.exists(SAMPLE_CSV):
        raise FileNotFoundError(
            f"no bundled CSV at {SAMPLE_CSV} and live fetch unavailable. "
            "Run once with network access to populate it."
        )
    rets = pd.read_csv(SAMPLE_CSV, index_col=0)
    cols = [t for t in tickers if t in rets.columns]
    if not cols:
        cols = list(rets.columns)
    return cols, rets[cols], "csv"


def moments(returns_df):
    """Expected-return vector mu and covariance Sigma (annualized ~252 days)."""
    mu = returns_df.mean().to_numpy() * 252.0
    sigma = returns_df.cov().to_numpy() * 252.0
    return mu, sigma


# ---------------------------------------------------------------------------
# 2. ENCODE  (QUBO and Ising)
# ---------------------------------------------------------------------------
def build_qubo(mu, sigma, K, q=0.5, penalty=None):
    """Mean-variance cardinality selection as a QUBO we MINIMIZE.

    We want to MAXIMIZE   q * mu^T x - x^T Sigma x   subject to sum x = K.
    Enforce the constraint with a penalty P*(sum x - K)^2 and flip the sign so
    the solver minimizes a cost:

        C(x) = -q*mu^T x + x^T Sigma x + P*(sum_i x_i - K)^2

    Expand the penalty:
        P*(sum x - K)^2 = P*[ sum_i x_i + 2*sum_{i<j} x_i x_j - 2K*sum_i x_i + K^2 ]
    (using x_i^2 = x_i for binaries).

    Returns (Q, c) with C(x) = x^T Q x + c, Q symmetric, diagonal carries the
    linear terms (since x_i^2 = x_i).
    """
    n = len(mu)
    if penalty is None:
        # Make the penalty just-binding: a hair above the per-asset objective
        # swing, so the cardinality constraint dominates the objective but the
        # cost landscape stays smooth enough for shallow QAOA to concentrate on
        # the feasible shell. A huge penalty flattens the QAOA distribution and
        # the optimum gets lost in near-uniform noise.
        obj_scale = float(
            np.abs(q * mu).max()
            + np.abs(np.diag(sigma)).max()
            + np.abs(sigma).max() * n
        )
        penalty = obj_scale

    Q = np.zeros((n, n))
    # Objective: -q*mu^T x  -> linear (diagonal);  x^T Sigma x -> full quadratic.
    for i in range(n):
        Q[i, i] += -q * mu[i]
    Q += sigma.copy()  # x^T Sigma x

    # Penalty terms.
    P = penalty
    for i in range(n):
        Q[i, i] += P * (1.0 - 2.0 * K)  # P*x_i - 2PK*x_i
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += P  # P*2*x_i x_j split symmetrically
            Q[j, i] += P
    const = P * (K ** 2)
    return Q, const


def qubo_cost(Q, const, x):
    """C(x) = x^T Q x + const for a binary vector x."""
    x = np.asarray(x, dtype=float)
    return float(x @ Q @ x + const)


def qubo_to_ising(Q, const):
    """Map x_i = (1 - z_i)/2 (z in {+1,-1}) to an Ising model.

    Returns (h, J, offset) such that
        C(x) = offset + sum_i h_i z_i + sum_{i<j} J_ij z_i z_j .

    h_i is the local Z field, J_ij the ZZ coupling.
    """
    n = Q.shape[0]
    # Work with linear a_i (diagonal) and quadratic b_ij (i<j, symmetric halves).
    a = np.array([Q[i, i] for i in range(n)])
    b = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            b[i, j] = Q[i, j] + Q[j, i]  # total coefficient on x_i x_j

    h = np.zeros(n)
    J = np.zeros((n, n))
    offset = const

    # Linear: a_i x_i = a_i*(1 - z_i)/2 = a_i/2 - (a_i/2) z_i
    for i in range(n):
        offset += a[i] / 2.0
        h[i] += -a[i] / 2.0

    # Quadratic: b_ij x_i x_j = b_ij*(1-z_i)(1-z_j)/4
    #   = b/4 * (1 - z_i - z_j + z_i z_j)
    for i in range(n):
        for j in range(i + 1, n):
            bij = b[i, j]
            offset += bij / 4.0
            h[i] += -bij / 4.0
            h[j] += -bij / 4.0
            J[i, j] += bij / 4.0
    return h, J, offset


def ising_energy(h, J, offset, z):
    """Energy of a spin configuration z in {+1,-1}^n."""
    z = np.asarray(z, dtype=float)
    n = len(z)
    e = offset + float(h @ z)
    for i in range(n):
        for j in range(i + 1, n):
            e += J[i, j] * z[i] * z[j]
    return e


def bits_to_spins(x):
    """x_i in {0,1} -> z_i in {+1,-1} via z = 1 - 2x  (x=0->+1, x=1->-1)."""
    return 1 - 2 * np.asarray(x, dtype=int)


# ---------------------------------------------------------------------------
# 3. QAOA CIRCUIT -> QASM
# ---------------------------------------------------------------------------
def qaoa_qasm(h, J, gammas, betas):
    """Emit an OpenQASM 2.0 QAOA circuit for the Ising (h, J).

    Cost layer for angle gamma:
        single-qubit  rz(2*gamma*h_i) q[i]
        ZZ term       cx q[i],q[j]; rz(2*gamma*J_ij) q[j]; cx q[i],q[j]
    Mixer for angle beta:  rx(2*beta) on every qubit.
    Initial state: h on every qubit (uniform superposition).
    """
    n = len(h)
    p = len(gammas)
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];"]
    for i in range(n):
        lines.append(f"h q[{i}];")
    for layer in range(p):
        g, b = gammas[layer], betas[layer]
        # Cost: Z fields.
        for i in range(n):
            if abs(h[i]) > 1e-15:
                lines.append(f"rz({2.0 * g * h[i]:.12g}) q[{i}];")
        # Cost: ZZ couplings.
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-15:
                    lines.append(f"cx q[{i}], q[{j}];")
                    lines.append(f"rz({2.0 * g * J[i, j]:.12g}) q[{j}];")
                    lines.append(f"cx q[{i}], q[{j}];")
        # Mixer.
        for i in range(n):
            lines.append(f"rx({2.0 * b:.12g}) q[{i}];")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 4. SIMULATE  (bridge to quasar)
# ---------------------------------------------------------------------------
def run_quasar(qasm_text, n):
    """Run quasar on QASM text; return a probability array of length 2^n.

    quasar `run` (no --shots) prints lines '<MSB-left bitstring>  <prob>' for
    every basis state with prob > 1e-9. We parse them into a dense vector
    indexed by the little-endian integer (qubit 0 = LSB), matching the encoding.
    """
    if not os.path.exists(QUASAR_BIN):
        raise FileNotFoundError(
            f"quasar binary not found at {QUASAR_BIN}. Build it first: "
            "cmake -S . -B build && cmake --build build"
        )
    with tempfile.NamedTemporaryFile("w", suffix=".qasm", delete=False) as f:
        f.write(qasm_text)
        path = f.name
    try:
        out = subprocess.run(
            [QUASAR_BIN, "run", path],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    finally:
        os.unlink(path)

    probs = np.zeros(1 << n)
    for line in out.strip().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        label, pstr = parts
        # label is MSB-left: label[0] is qubit n-1, label[-1] is qubit 0.
        idx = int(label, 2)  # MSB-left binary string -> integer directly.
        probs[idx] = float(pstr)
    return probs


# ---------------------------------------------------------------------------
# 5. OPTIMIZE ANGLES
# ---------------------------------------------------------------------------
def expected_cost(probs, Q, const, n):
    """E = sum_x P(x) C(x), iterating only over states with nonzero prob."""
    e = 0.0
    nz = np.nonzero(probs)[0]
    for idx in nz:
        x = [(idx >> q) & 1 for q in range(n)]  # qubit q -> bit q (little-endian)
        e += probs[idx] * qubo_cost(Q, const, x)
    return e


def optimize_angles(h, J, Q, const, n, p=1, seed=0):
    """Optimize (gammas, betas) to minimize E. COBYLA with a grid warm start.

    Returns (best_params, best_E, n_evals).
    """
    evals = {"n": 0}

    def objective(params):
        gammas = params[:p]
        betas = params[p:]
        qasm = qaoa_qasm(h, J, gammas, betas)
        probs = run_quasar(qasm, n)
        evals["n"] += 1
        return expected_cost(probs, Q, const, n)

    # Grid warm start over (gamma, beta) for the first layer to seed COBYLA.
    rng = np.random.default_rng(seed)
    best_x0, best_e0 = None, math.inf
    grid = np.linspace(0.0, math.pi, 6)
    for g in grid:
        for b in np.linspace(0.0, math.pi / 2, 5):
            x0 = np.array([g] * p + [b] * p)
            e = objective(x0)
            if e < best_e0:
                best_e0, best_x0 = e, x0

    # Local refinement with COBYLA.
    try:
        from scipy.optimize import minimize

        res = minimize(
            objective,
            best_x0,
            method="COBYLA",
            options={"maxiter": 60, "rhobeg": 0.4, "tol": 1e-4},
        )
        if res.fun < best_e0:
            return res.x, float(res.fun), evals["n"]
    except Exception as e:  # noqa: BLE001
        print(f"[optimize] scipy refinement skipped ({e!r})")
    return best_x0, best_e0, evals["n"]


# ---------------------------------------------------------------------------
# 6. RESULT  (brute force + QAOA basket)
# ---------------------------------------------------------------------------
def brute_force(Q, const, n, K):
    """Brute-force the minimum-cost basket with exactly K assets selected."""
    best_x, best_c = None, math.inf
    for combo in itertools.combinations(range(n), K):
        x = [0] * n
        for i in combo:
            x[i] = 1
        c = qubo_cost(Q, const, x)
        if c < best_c:
            best_c, best_x = c, x
    return best_x, best_c


def qaoa_basket(probs, n, K):
    """Most-probable FEASIBLE (exactly-K) bitstring; fall back to top overall."""
    order = np.argsort(probs)[::-1]
    for idx in order:
        if probs[idx] <= 0:
            break
        x = [(idx >> q) & 1 for q in range(n)]
        if sum(x) == K:
            return x, float(probs[idx])
    # No feasible state had positive prob; return the global top.
    idx = int(order[0])
    x = [(idx >> q) & 1 for q in range(n)]
    return x, float(probs[idx])


def basket_names(x, tickers):
    return [tickers[i] for i in range(len(x)) if x[i]]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_portfolio(tickers, mu, sigma, K, q=0.5, p=1, verbose=True):
    """End-to-end: encode -> QAOA on quasar -> compare to brute force.

    Returns a dict with all artifacts (usable from tests).
    """
    n = len(mu)
    Q, const = build_qubo(mu, sigma, K, q=q)
    h, J, offset = qubo_to_ising(Q, const)

    params, qaoa_E, n_evals = optimize_angles(h, J, Q, const, n, p=p)
    gammas, betas = params[:p], params[p:]
    probs = run_quasar(qaoa_qasm(h, J, gammas, betas), n)

    qx, qprob = qaoa_basket(probs, n, K)
    bx, bc = brute_force(Q, const, n, K)
    qc = qubo_cost(Q, const, qx)

    matched = (qx == bx)
    result = {
        "n": n,
        "K": K,
        "p": p,
        "tickers": tickers,
        "qaoa_basket": qx,
        "qaoa_cost": qc,
        "qaoa_prob": qprob,
        "qaoa_expected_cost": qaoa_E,
        "brute_basket": bx,
        "brute_cost": bc,
        "matched": matched,
        "n_evals": n_evals,
        "gammas": list(gammas),
        "betas": list(betas),
    }

    if verbose:
        print("=" * 68)
        print("QAOA PORTFOLIO OPTIMIZATION on the quasar simulator")
        print("=" * 68)
        print(DISCLAIMER)
        print("-" * 68)
        print(f"assets (N={n}): {', '.join(tickers)}")
        print(f"select exactly K={K}, q-tilt={q}, QAOA layers p={p}, "
              f"quasar evals={n_evals}")
        print(f"angles: gammas={[round(g,3) for g in gammas]} "
              f"betas={[round(b,3) for b in betas]}")
        print("-" * 68)
        print(f"QAOA basket       : {basket_names(qx, tickers)}")
        print(f"  objective C(x)  : {qc:.6f}   (most-probable feasible, "
              f"p={qprob:.4f})")
        print(f"  QAOA E[C]       : {qaoa_E:.6f}")
        print(f"brute-force basket: {basket_names(bx, tickers)}")
        print(f"  objective C(x)  : {bc:.6f}")
        print("-" * 68)
        print(f"MATCH: {'YES -- QAOA found the classical optimum' if matched else 'NO -- QAOA basket differs (see caveats)'}")
        print(f"cost gap (QAOA - brute): {qc - bc:.6e}")
        print("=" * 68)
    return result


def main():
    ap = argparse.ArgumentParser(description="QAOA portfolio optimization on quasar")
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--k", type=int, default=4, help="cardinality (assets to pick)")
    ap.add_argument("--q", type=float, default=0.5, help="return-tilt weight")
    ap.add_argument("--p", type=int, default=2, help="QAOA layers")
    ap.add_argument("--period", default="2y")
    ap.add_argument("--offline", action="store_true",
                    help="skip yfinance; use the bundled CSV")
    args = ap.parse_args()

    tickers, rets, source = fetch_returns(
        args.tickers, period=args.period, offline=args.offline
    )
    print(f"[data] source={source}  rows={rets.shape[0]}  assets={list(rets.columns)}")
    mu, sigma = moments(rets)
    run_portfolio(tickers, mu, sigma, args.k, q=args.q, p=args.p)


if __name__ == "__main__":
    main()
