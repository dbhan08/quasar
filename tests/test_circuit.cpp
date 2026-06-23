#include <gtest/gtest.h>

#include <cmath>

#include "quasar/circuit.hpp"
#include "quasar/gates.hpp"
#include "quasar/statevector.hpp"

using namespace quasar;

namespace {
constexpr double kTol = 1e-12;
constexpr double kInvSqrt2 = 0.7071067811865475244;
}  // namespace

TEST(Circuit, BellState) {
  // H(0), CX(0,1) -> (|00> + |11>)/sqrt2.
  Circuit c(2);
  c.h(0).cx(0, 1);
  StateVector sv(2);
  apply(c, sv);
  EXPECT_NEAR(std::abs(sv.data()[0] - cd(kInvSqrt2, 0)), 0.0, kTol);  // |00>
  EXPECT_NEAR(std::abs(sv.data()[1]), 0.0, kTol);
  EXPECT_NEAR(std::abs(sv.data()[2]), 0.0, kTol);
  EXPECT_NEAR(std::abs(sv.data()[3] - cd(kInvSqrt2, 0)), 0.0, kTol);  // |11>
}

TEST(Circuit, GHZThree) {
  // H(0), CX(0,1), CX(1,2) -> (|000> + |111>)/sqrt2.
  Circuit c(3);
  c.h(0).cx(0, 1).cx(1, 2);
  StateVector sv(3);
  apply(c, sv);
  for (std::size_t i = 0; i < 8; ++i) {
    if (i == 0 || i == 7)
      EXPECT_NEAR(std::abs(sv.data()[i] - cd(kInvSqrt2, 0)), 0.0, kTol);
    else
      EXPECT_NEAR(std::abs(sv.data()[i]), 0.0, kTol);
  }
}

TEST(Circuit, CnotLittleEndianControlOnTarget) {
  // Prepare |01> (qubit0=1), CX(0,1) -> qubit1 flips -> |11> = index 3.
  Circuit c(2);
  c.x(0).cx(0, 1);
  StateVector sv(2);
  apply(c, sv);
  EXPECT_NEAR(std::abs(sv.data()[3] - cd(1, 0)), 0.0, kTol);
}

TEST(Circuit, CnotControlZeroNoEffect) {
  // |00>, CX(0,1) -> unchanged |00>.
  Circuit c(2);
  c.cx(0, 1);
  StateVector sv(2);
  apply(c, sv);
  EXPECT_NEAR(std::abs(sv.data()[0] - cd(1, 0)), 0.0, kTol);
}

TEST(Circuit, Swap) {
  // |01> swap(0,1) -> |10> = index 2.
  Circuit c(2);
  c.x(0).swap(0, 1);
  StateVector sv(2);
  apply(c, sv);
  EXPECT_NEAR(std::abs(sv.data()[2] - cd(1, 0)), 0.0, kTol);
}

TEST(Circuit, CZPhase) {
  // |11> CZ -> -|11>.
  Circuit c(2);
  c.x(0).x(1).cz(0, 1);
  StateVector sv(2);
  apply(c, sv);
  EXPECT_NEAR(std::abs(sv.data()[3] - cd(-1, 0)), 0.0, kTol);
}

TEST(Circuit, RotationIdentityRZ2PiIsGlobalPhase) {
  // RZ(2pi) = diag(e^{-i pi}, e^{i pi}) = -I (global phase).
  StateVector sv(1);
  sv.apply1(0, gates::H());
  auto before = sv.data();
  sv.apply1(0, gates::RZ(2 * M_PI));
  for (std::size_t i = 0; i < 2; ++i)
    EXPECT_NEAR(std::abs(sv.data()[i] - cd(-1, 0) * before[i]), 0.0, kTol);
}

TEST(Circuit, RXPiIsMinusiX) {
  // RX(pi)|0> = -i|1>.
  StateVector sv(1);
  sv.apply1(0, gates::RX(M_PI));
  EXPECT_NEAR(std::abs(sv.data()[0]), 0.0, kTol);
  EXPECT_NEAR(std::abs(sv.data()[1] - cd(0, -1)), 0.0, kTol);
}

TEST(Circuit, RYPiIsX) {
  // RY(pi)|0> = |1>.
  StateVector sv(1);
  sv.apply1(0, gates::RY(M_PI));
  EXPECT_NEAR(std::abs(sv.data()[1] - cd(1, 0)), 0.0, kTol);
}

TEST(Circuit, SGateIsSqrtZ) {
  // S applied twice == Z.
  StateVector sv(1);
  sv.apply1(0, gates::X());  // |1>
  sv.apply1(0, gates::S());
  sv.apply1(0, gates::S());
  EXPECT_NEAR(std::abs(sv.data()[1] - cd(-1, 0)), 0.0, kTol);
}

TEST(Circuit, ControlledRZ) {
  // Controlled-RZ leaves |0_c> alone, applies phase when control=1.
  Circuit c(2);
  c.x(0).x(1);  // |11>
  c.controlled("rz", 0, 1, M_PI);  // crz on target qubit1
  StateVector sv(2);
  apply(c, sv);
  // RZ(pi) on |1> gives e^{i pi/2}|1>; index 3 amplitude = e^{i pi/2}.
  EXPECT_NEAR(std::abs(sv.data()[3] - std::exp(cd(0, M_PI / 2))), 0.0, kTol);
}

TEST(Circuit, ApplyMultiMatchesCnot) {
  // applyMulti with a 4x4 CNOT (control=qubit0, target=qubit1) should match.
  StateVector sv(2);
  sv.apply1(0, gates::H());
  StateVector sv2 = sv;
  sv.applyControlled1(0, 1, gates::X());
  // 4x4 CNOT in little-endian where targets={0,1}: basis order |b1 b0> with
  // gathered bit j=0 -> qubit0, j=1 -> qubit1. Index s = b0 + 2*b1.
  // CNOT(control=q0,target=q1): if b0==1 flip b1.
  std::vector<cd> U(16, cd(0, 0));
  auto set = [&](int r, int col) { U[r * 4 + col] = cd(1, 0); };
  set(0, 0);  // 00 -> 00
  set(1, 3);  // 01(s=1,b0=1,b1=0) -> b1 flips -> b1=1,b0=1 -> s=3
  set(2, 2);  // 10 -> 10
  set(3, 1);  // 11 -> b1 flips -> b1=0,b0=1 -> s=1
  // Wait: build mapping carefully below in test body via direct compare.
  sv2.applyMulti({0, 1}, U);
  for (std::size_t i = 0; i < 4; ++i)
    EXPECT_NEAR(std::abs(sv.data()[i] - sv2.data()[i]), 0.0, kTol);
}
