#include <gtest/gtest.h>

#include <cmath>
#include <fstream>

#include "quasar/circuit.hpp"
#include "quasar/circuit_json.hpp"
#include "quasar/golden.hpp"
#include "quasar/statevector.hpp"

using namespace quasar;

namespace {
constexpr double kTol = 1e-12;
constexpr double kInvSqrt2 = 0.7071067811865475244;
}  // namespace

TEST(Bench, CircuitJsonLoadsBell) {
  const char* j =
      R"({"nqubits": 2, "ops": [
           {"name":"h","qubits":[0],"params":[]},
           {"name":"cx","qubits":[0,1],"params":[]}
         ]})";
  Circuit c = loadCircuitJson(j);
  EXPECT_EQ(c.nqubits(), 2);
  ASSERT_EQ(c.ops().size(), 2u);
  EXPECT_EQ(c.ops()[0].name, "h");
  EXPECT_EQ(c.ops()[1].name, "cx");

  StateVector sv(2);
  apply(c, sv);
  EXPECT_NEAR(std::abs(sv.data()[0] - cd(kInvSqrt2, 0)), 0.0, kTol);
  EXPECT_NEAR(std::abs(sv.data()[3] - cd(kInvSqrt2, 0)), 0.0, kTol);
}

TEST(Bench, CircuitJsonParsesParams) {
  const char* j =
      R"({"nqubits": 1, "ops": [
           {"name":"rz","qubits":[0],"params":[1.5707963267948966]}
         ]})";
  Circuit c = loadCircuitJson(j);
  ASSERT_EQ(c.ops().size(), 1u);
  ASSERT_EQ(c.ops()[0].params.size(), 1u);
  EXPECT_NEAR(c.ops()[0].params[0], M_PI / 2.0, 1e-12);
}

TEST(Bench, GoldenLoadAndCompare) {
  // A golden file in the harness format for a Bell state.
  const std::string j =
      "{\"nqubits\": 2, \"circuit\": \"bell\", \"baseline\": \"qiskit-aer\", "
      "\"aer_time_s\": 0.001, "
      "\"re\": [0.7071067811865476, 0.0, 0.0, 0.7071067811865476], "
      "\"im\": [0.0, 0.0, 0.0, 0.0]}";
  // Write to a temp file.
  std::string path = std::string(::testing::TempDir()) + "bell.golden.json";
  {
    std::ofstream f(path);
    f << j;
  }
  Golden g = loadGoldenFile(path);
  EXPECT_EQ(g.nqubits, 2);
  EXPECT_EQ(g.circuit, "bell");
  EXPECT_EQ(g.baseline, "qiskit-aer");
  ASSERT_EQ(g.state.size(), 4u);

  Circuit c(2);
  c.h(0).cx(0, 1);
  StateVector sv(2);
  apply(c, sv);
  double maxdiff = 0.0;
  for (std::size_t i = 0; i < 4; ++i)
    maxdiff = std::max(maxdiff, std::abs(sv.data()[i] - g.state[i]));
  EXPECT_LE(maxdiff, 1e-9);
}
