#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "quasar/gates.hpp"
#include "quasar/statevector.hpp"

using namespace quasar;

namespace {
constexpr double kTol = 1e-12;
constexpr double kInvSqrt2 = 0.7071067811865475244;
}  // namespace

TEST(StateVector, InitToZeroState) {
  StateVector sv(3);
  EXPECT_EQ(sv.dim(), 8u);
  EXPECT_NEAR(std::abs(sv.data()[0] - cd(1, 0)), 0.0, kTol);
  for (std::size_t i = 1; i < sv.dim(); ++i)
    EXPECT_NEAR(std::abs(sv.data()[i]), 0.0, kTol);
}

TEST(StateVector, XFlipsZeroToOne) {
  StateVector sv(1);
  sv.apply1(0, gates::X());
  EXPECT_NEAR(std::abs(sv.data()[0]), 0.0, kTol);
  EXPECT_NEAR(std::abs(sv.data()[1] - cd(1, 0)), 0.0, kTol);
}

TEST(StateVector, HadamardMakesPlus) {
  StateVector sv(1);
  sv.apply1(0, gates::H());
  EXPECT_NEAR(std::abs(sv.data()[0] - cd(kInvSqrt2, 0)), 0.0, kTol);
  EXPECT_NEAR(std::abs(sv.data()[1] - cd(kInvSqrt2, 0)), 0.0, kTol);
  auto p = sv.probabilities();
  EXPECT_NEAR(p[0], 0.5, kTol);
  EXPECT_NEAR(p[1], 0.5, kTol);
}

TEST(StateVector, ZPhaseOnOne) {
  StateVector sv(1);
  sv.apply1(0, gates::X());  // |1>
  sv.apply1(0, gates::Z());  // -|1>
  EXPECT_NEAR(std::abs(sv.data()[1] - cd(-1, 0)), 0.0, kTol);
}

TEST(StateVector, HHIsIdentity) {
  StateVector sv(1);
  sv.apply1(0, gates::H());
  sv.apply1(0, gates::H());
  EXPECT_NEAR(std::abs(sv.data()[0] - cd(1, 0)), 0.0, kTol);
  EXPECT_NEAR(std::abs(sv.data()[1]), 0.0, kTol);
}

TEST(StateVector, NormalizationPreserved) {
  StateVector sv(4);
  sv.apply1(0, gates::H());
  sv.apply1(1, gates::H());
  sv.apply1(2, gates::X());
  sv.apply1(3, gates::H());
  sv.apply1(1, gates::Z());
  EXPECT_NEAR(sv.normSquared(), 1.0, kTol);
}

TEST(StateVector, GateOnSpecificQubitLittleEndian) {
  // Apply X to qubit 2 of |000> -> index 4 (b2=1).
  StateVector sv(3);
  sv.apply1(2, gates::X());
  EXPECT_NEAR(std::abs(sv.data()[4] - cd(1, 0)), 0.0, kTol);
}

TEST(StateVector, SampleDeterministicForBasisState) {
  StateVector sv(2);
  sv.apply1(0, gates::X());  // |01> = index 1
  std::mt19937_64 rng(42);
  for (int i = 0; i < 10; ++i) EXPECT_EQ(sv.sample(rng), 1u);
}

TEST(StateVector, SampleFairForPlus) {
  StateVector sv(1);
  sv.apply1(0, gates::H());
  std::mt19937_64 rng(123);
  int ones = 0;
  const int N = 20000;
  for (int i = 0; i < N; ++i) ones += static_cast<int>(sv.sample(rng));
  double frac = static_cast<double>(ones) / N;
  EXPECT_NEAR(frac, 0.5, 0.02);
}
