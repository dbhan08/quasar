#include <gtest/gtest.h>

#include <cmath>
#include <complex>

#include "quasar/algorithms.hpp"
#include "quasar/circuit.hpp"
#include "quasar/gates.hpp"
#include "quasar/qasm.hpp"
#include "quasar/statevector.hpp"

using namespace quasar;

namespace {
constexpr double kTol = 1e-10;
}

// QFT on a computational basis state |x> must produce the analytic DFT:
//   amp[y] = (1/sqrt(N)) exp(2*pi*i*x*y/N).
TEST(Algorithms, QFTMatchesAnalyticDFT) {
  for (int n = 1; n <= 4; ++n) {
    const std::size_t N = std::size_t{1} << n;
    for (std::size_t x = 0; x < N; ++x) {
      StateVector sv(n);
      // Prepare |x> by flipping bits.
      for (int q = 0; q < n; ++q)
        if ((x >> q) & 1ULL) sv.apply1(q, gates::X());
      apply(qft(n), sv);
      for (std::size_t y = 0; y < N; ++y) {
        double ang = 2.0 * M_PI * static_cast<double>(x) *
                     static_cast<double>(y) / static_cast<double>(N);
        cd expected = std::exp(cd(0, ang)) / std::sqrt(static_cast<double>(N));
        EXPECT_NEAR(std::abs(sv.data()[y] - expected), 0.0, kTol)
            << "n=" << n << " x=" << x << " y=" << y;
      }
    }
  }
}

TEST(Algorithms, GHZExactState) {
  const double inv = 1.0 / std::sqrt(2.0);
  for (int n = 2; n <= 5; ++n) {
    StateVector sv(n);
    apply(ghz(n), sv);
    const std::size_t N = std::size_t{1} << n;
    for (std::size_t i = 0; i < N; ++i) {
      if (i == 0 || i == N - 1)
        EXPECT_NEAR(std::abs(sv.data()[i] - cd(inv, 0)), 0.0, kTol);
      else
        EXPECT_NEAR(std::abs(sv.data()[i]), 0.0, kTol);
    }
  }
}

TEST(Algorithms, GroverAmplifiesMarked) {
  for (int n = 3; n <= 4; ++n) {
    const std::size_t N = std::size_t{1} << n;
    for (std::uint64_t marked = 0; marked < N; ++marked) {
      StateVector sv(n);
      apply(grover(n, marked), sv);
      auto p = sv.probabilities();
      EXPECT_GT(p[marked], 0.5) << "n=" << n << " marked=" << marked
                                << " prob=" << p[marked];
    }
  }
}

TEST(Algorithms, QAOAStaysNormalized) {
  std::vector<double> g = {0.3, 0.7};
  std::vector<double> b = {0.5, 0.2};
  StateVector sv(4);
  apply(qaoa_ansatz(4, g, b), sv);
  EXPECT_NEAR(sv.normSquared(), 1.0, kTol);
}

TEST(Qasm, ParsesBellAndMatchesBuilder) {
  const char* src =
      "OPENQASM 2.0;\n"
      "include \"qelib1.inc\";\n"
      "qreg q[2];\n"
      "h q[0];\n"
      "cx q[0], q[1];\n";
  Circuit fromQasm = parseQasm(src);
  StateVector a(2);
  apply(fromQasm, a);

  Circuit ref(2);
  ref.h(0).cx(0, 1);
  StateVector b(2);
  apply(ref, b);

  for (std::size_t i = 0; i < 4; ++i)
    EXPECT_NEAR(std::abs(a.data()[i] - b.data()[i]), 0.0, kTol);
}

TEST(Qasm, ParsesRotationAngles) {
  const char* src =
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "rx(pi) q[0];\n";
  Circuit c = parseQasm(src);
  ASSERT_EQ(c.ops().size(), 1u);
  EXPECT_EQ(c.ops()[0].name, "rx");
  ASSERT_EQ(c.ops()[0].params.size(), 1u);
  EXPECT_NEAR(c.ops()[0].params[0], M_PI, 1e-12);

  StateVector sv(1);
  apply(c, sv);
  // RX(pi)|0> = -i|1>.
  EXPECT_NEAR(std::abs(sv.data()[1] - cd(0, -1)), 0.0, kTol);
}

TEST(Qasm, RoundTripQAOAGatesParse) {
  const char* src =
      "OPENQASM 2.0;\n"
      "qreg q[3];\n"
      "h q[0];\n"
      "h q[1];\n"
      "h q[2];\n"
      "cx q[0], q[1];\n"
      "rz(0.6) q[1];\n"
      "cx q[0], q[1];\n"
      "rx(1.0) q[0];\n";
  Circuit c = parseQasm(src);
  EXPECT_EQ(c.nqubits(), 3);
  EXPECT_EQ(c.ops().size(), 7u);
  StateVector sv(3);
  apply(c, sv);
  EXPECT_NEAR(sv.normSquared(), 1.0, kTol);
}
