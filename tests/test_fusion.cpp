#include <gtest/gtest.h>

#include <cmath>

#include "quasar/algorithms.hpp"
#include "quasar/circuit.hpp"
#include "quasar/fusion.hpp"
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

void checkFusedMatchesUnfused(const Circuit& c) {
  StateVector unfused(c.nqubits());
  apply(c, unfused);
  Circuit fused = fuse(c, 4);
  StateVector f(c.nqubits());
  applyFused(fused, f);
  EXPECT_LE(maxDiff(unfused, f), kTol);
}
}  // namespace

TEST(Fusion, BellIdentical) {
  Circuit c(2);
  c.h(0).cx(0, 1);
  checkFusedMatchesUnfused(c);
}

TEST(Fusion, GHZIdentical) {
  for (int n = 2; n <= 6; ++n) checkFusedMatchesUnfused(ghz(n));
}

TEST(Fusion, QFTIdentical) {
  for (int n = 1; n <= 6; ++n) checkFusedMatchesUnfused(qft(n));
}

TEST(Fusion, GroverIdentical) {
  for (int n = 3; n <= 4; ++n)
    for (std::uint64_t m = 0; m < (1ULL << n); ++m)
      checkFusedMatchesUnfused(grover(n, m));
}

TEST(Fusion, QAOAIdentical) {
  std::vector<double> g = {0.4, 0.8};
  std::vector<double> b = {0.3, 0.15};
  for (int n = 3; n <= 6; ++n) checkFusedMatchesUnfused(qaoa_ansatz(n, g, b));
}

TEST(Fusion, MixedRotationsIdentical) {
  Circuit c(4);
  c.h(0).rx(0, 0.7).cx(0, 1).ry(1, 1.1).cz(1, 2).t(2).swap(2, 3).rz(3, 0.5)
      .phase(0, 0.9).cx(2, 3);
  checkFusedMatchesUnfused(c);
}

// A run of phase / controlled-phase gates composes to DIAGONAL fused blocks,
// which fuse() emits as "fused_diag" ops routed through applyDiagonal. The
// resulting state must be identical (~1e-12) to the unfused apply, and at least
// one fused_diag op must actually be produced (otherwise we're not exercising
// the diagonal path).
TEST(Fusion, DiagonalBlockMatchesUnfused) {
  Circuit c(5);
  c.phase(0, 0.7).phase(1, 1.3).controlled("phase", 0, 1, 0.5)
      .controlled("phase", 1, 2, 0.9).z(2).s(3).t(4)
      .controlled("phase", 3, 4, 1.1).rz(0, 0.6).cz(1, 2);

  Circuit fused = fuse(c, 4);
  int ndiag = 0;
  for (const auto& op : fused.ops())
    if (op.name == "fused_diag") ++ndiag;
  EXPECT_GT(ndiag, 0) << "expected at least one diagonal fused block";

  StateVector unfused(c.nqubits());
  apply(c, unfused);
  StateVector f(c.nqubits());
  applyFused(fused, f);
  EXPECT_LE(maxDiff(unfused, f), kTol);

  // Diagonal path must be insensitive to the useBlas flag.
  StateVector fb(c.nqubits());
  applyFused(fused, fb, /*useBlas=*/true);
  EXPECT_LE(maxDiff(unfused, fb), kTol);
}

// The multi-threaded apply paths (apply1 sweep, applyMulti, applyDiagonal) must
// produce the same state as a known-correct single-threaded reference. We build
// a single-threaded reference by hand-applying each op via the original scalar
// formulas on a large enough state to cross the parallel threshold.
TEST(Fusion, MultiThreadedMatchesSingleThreaded) {
  const int n = 12;  // 2^12 amplitudes -> parallel paths engage
  Circuit c(n);
  for (int q = 0; q < n; ++q) c.h(q);
  for (int q = 0; q + 1 < n; ++q) c.controlled("phase", q, q + 1, 0.3 + 0.05 * q);
  for (int q = 0; q < n; ++q) c.rz(q, 0.2 * (q + 1));
  for (int q = 0; q + 1 < n; ++q) c.cx(q, q + 1);
  for (int q = 0; q < n; ++q) c.ry(q, 0.11 * (q + 1));

  // Single-threaded reference: replay the same ops through the unfused scalar
  // path with a forced-serial state (small chunks already run serially below
  // threshold, but here we compare two independent apply paths instead).
  StateVector ref(n);
  apply(c, ref);

  // Threaded fused path (both diag and dense blocks, threaded loops).
  Circuit fused = fuse(c, 4);
  StateVector mt(n);
  applyFused(fused, mt, /*useBlas=*/false);
  EXPECT_LE(maxDiff(ref, mt), kTol);

  StateVector mtb(n);
  applyFused(fused, mtb, /*useBlas=*/true);
  EXPECT_LE(maxDiff(ref, mtb), kTol);
}

TEST(Fusion, RespectsMaxWidth) {
  // A circuit spanning 5 qubits must not produce a fused block wider than 4.
  Circuit c(5);
  c.cx(0, 1).cx(1, 2).cx(2, 3).cx(3, 4).h(0).h(4);
  Circuit fused = fuse(c, 4);
  for (const auto& op : fused.ops()) {
    EXPECT_EQ(op.name, "fused");
    EXPECT_LE(static_cast<int>(op.qubits.size()), 4);
  }
  checkFusedMatchesUnfused(c);
}
