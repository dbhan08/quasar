"""Tests for the QAOA portfolio-optimization capstone.

Run: .venv/bin/python -m pytest capstone/
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import portfolio_qaoa as pq  # noqa: E402

QUASAR_BIN = pq.QUASAR_BIN
HAVE_QUASAR = os.path.exists(QUASAR_BIN)


# ---------------------------------------------------------------------------
# A hand-verifiable N=4 portfolio.
# ---------------------------------------------------------------------------
def hand_portfolio():
    """Four assets; assets 0 and 1 have high return + low variance and are
    uncorrelated, assets 2 and 3 are poor (low/negative return, high variance).

    With q*mu reward and quadratic risk, the best K=2 basket by hand is {0, 1}:
    they have the highest returns and, being uncorrelated with low variance,
    the lowest pairwise risk. We verify brute force agrees, then check QAOA.
    """
    mu = np.array([0.30, 0.28, 0.02, -0.05])
    # Diagonal-dominant covariance: low variance for the good pair.
    sigma = np.array([
        [0.04, 0.00, 0.01, 0.01],
        [0.00, 0.05, 0.01, 0.01],
        [0.01, 0.01, 0.20, 0.05],
        [0.01, 0.01, 0.05, 0.25],
    ])
    return mu, sigma


def test_handcrafted_optimum_is_asset_0_and_1():
    """Brute force returns the hand-known optimal K=2 basket {0,1}."""
    mu, sigma = hand_portfolio()
    Q, const = pq.build_qubo(mu, sigma, K=2, q=1.0)
    bx, bc = pq.brute_force(Q, const, n=4, K=2)
    assert bx == [1, 1, 0, 0], f"expected basket {{0,1}}, got {bx}"


@pytest.mark.skipif(not HAVE_QUASAR, reason="quasar binary not built")
def test_qaoa_matches_bruteforce_n4():
    """QAOA on quasar finds the hand-known optimum, and its expected cost is
    within tolerance of the brute-force optimum."""
    mu, sigma = hand_portfolio()
    res = pq.run_portfolio(
        ["A0", "A1", "A2", "A3"], mu, sigma, K=2, q=1.0, p=2, verbose=False
    )
    assert res["qaoa_basket"] == [1, 1, 0, 0], res["qaoa_basket"]
    assert res["matched"] is True
    # QAOA's most-probable feasible basket attains the brute-force cost exactly.
    assert abs(res["qaoa_cost"] - res["brute_cost"]) < 1e-9
    # Expected cost E[C] is finite and not absurdly far from the optimum: with a
    # tuned penalty and p=2 the distribution concentrates on feasible states.
    assert res["qaoa_expected_cost"] >= res["brute_cost"] - 1e-9
    assert res["qaoa_expected_cost"] < res["brute_cost"] + 5.0


# ---------------------------------------------------------------------------
# QUBO <-> Ising round-trip.
# ---------------------------------------------------------------------------
def test_qubo_ising_roundtrip_random():
    """For several random binary x, C(x) from the QUBO equals the Ising energy
    of the mapped spins z = 1 - 2x."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        n = rng.integers(3, 7)
        mu = rng.normal(0.1, 0.1, size=n)
        A = rng.normal(0, 0.1, size=(n, n))
        sigma = A @ A.T  # PSD covariance
        K = int(rng.integers(1, n))
        Q, const = pq.build_qubo(mu, sigma, K, q=rng.uniform(0.2, 1.0))
        h, J, offset = pq.qubo_to_ising(Q, const)
        for _ in range(8):
            x = rng.integers(0, 2, size=n)
            z = pq.bits_to_spins(x)
            c_qubo = pq.qubo_cost(Q, const, x)
            e_ising = pq.ising_energy(h, J, offset, z)
            assert abs(c_qubo - e_ising) < 1e-9, (c_qubo, e_ising)


def test_qubo_penalty_enforces_cardinality():
    """The minimum-cost basket has exactly K assets (penalty binds)."""
    mu, sigma = hand_portfolio()
    for K in (1, 2, 3):
        Q, const = pq.build_qubo(mu, sigma, K, q=1.0)
        # Global minimum over ALL 2^n baskets should have exactly K bits set.
        best_x, best_c = None, float("inf")
        for mask in range(1 << 4):
            x = [(mask >> i) & 1 for i in range(4)]
            c = pq.qubo_cost(Q, const, x)
            if c < best_c:
                best_c, best_x = c, x
        assert sum(best_x) == K, (K, best_x)


@pytest.mark.skipif(not HAVE_QUASAR, reason="quasar binary not built")
def test_quasar_probs_normalized():
    """The probability vector parsed from quasar sums to ~1."""
    mu, sigma = hand_portfolio()
    Q, const = pq.build_qubo(mu, sigma, K=2, q=1.0)
    h, J, _ = pq.qubo_to_ising(Q, const)
    qasm = pq.qaoa_qasm(h, J, gammas=[0.5, 0.3], betas=[0.4, 0.2])
    probs = pq.run_quasar(qasm, n=4)
    assert abs(probs.sum() - 1.0) < 1e-6, probs.sum()
