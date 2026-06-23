#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "quasar/algorithms.hpp"
#include "quasar/circuit.hpp"
#include "quasar/fusion.hpp"
#include "quasar/gates.hpp"
#include "quasar/statevector.hpp"

using namespace quasar;

namespace {
constexpr double kTol = 1e-12;

double maxDiff(const StateVector& a, const StateVector& b) {
  double m = 0.0;
  for (std::size_t i = 0; i < a.dim(); ++i)
    m = std::max(m, std::abs(a.data()[i] - b.data()[i]));
  return m;
}
}  // namespace

// applyMultiBlas must match applyMulti bit-for-bit (~1e-12) for random gates on
// random target sets.
TEST(Blas, MatchesScalarRandomGates) {
  std::mt19937_64 rng(7);
  std::normal_distribution<double> nd(0.0, 1.0);
  const int N = 8;

  for (int trial = 0; trial < 30; ++trial) {
    // Random initial state.
    StateVector seed(N);
    std::vector<cd>& d = seed.data();
    double norm = 0.0;
    for (auto& z : d) {
      z = cd(nd(rng), nd(rng));
      norm += std::norm(z);
    }
    norm = std::sqrt(norm);
    for (auto& z : d) z /= norm;

    // Random target set of size k (distinct qubits).
    int k = 1 + (rng() % 4);
    std::vector<int> all(N);
    for (int i = 0; i < N; ++i) all[i] = i;
    std::shuffle(all.begin(), all.end(), rng);
    std::vector<int> targets(all.begin(), all.begin() + k);

    const std::size_t K = std::size_t{1} << k;
    std::vector<cd> u(K * K);
    for (auto& z : u) z = cd(nd(rng), nd(rng));

    StateVector a = seed, b = seed;
    a.applyMulti(targets, u);
    b.applyMultiBlas(targets, u);
    EXPECT_LE(maxDiff(a, b), kTol) << "trial=" << trial << " k=" << k;
  }
}

TEST(Blas, FusedBlasMatchesUnfusedQFT) {
  for (int n = 1; n <= 6; ++n) {
    StateVector unfused(n);
    apply(qft(n), unfused);
    Circuit f = fuse(qft(n), 4);
    StateVector blas(n);
    applyFused(f, blas, /*useBlas=*/true);
    EXPECT_LE(maxDiff(unfused, blas), kTol) << "n=" << n;
  }
}

TEST(Blas, FusedBlasMatchesUnfusedGrover) {
  for (int n = 3; n <= 4; ++n)
    for (std::uint64_t m = 0; m < (1ULL << n); ++m) {
      StateVector unfused(n);
      apply(grover(n, m), unfused);
      Circuit f = fuse(grover(n, m), 4);
      StateVector blas(n);
      applyFused(f, blas, true);
      EXPECT_LE(maxDiff(unfused, blas), kTol);
    }
}

TEST(Blas, FusedBlasMatchesUnfusedQAOA) {
  std::vector<double> g = {0.4, 0.8};
  std::vector<double> b = {0.3, 0.15};
  for (int n = 3; n <= 6; ++n) {
    StateVector unfused(n);
    apply(qaoa_ansatz(n, g, b), unfused);
    Circuit f = fuse(qaoa_ansatz(n, g, b), 4);
    StateVector blas(n);
    applyFused(f, blas, true);
    EXPECT_LE(maxDiff(unfused, blas), kTol);
  }
}
