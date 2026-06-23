#include "quasar/gates.hpp"

#include <cmath>

namespace quasar {
namespace gates {

namespace {
constexpr double kInvSqrt2 = 0.7071067811865475244;  // 1/sqrt(2)
}

std::array<cd, 4> X() { return {cd(0, 0), cd(1, 0), cd(1, 0), cd(0, 0)}; }

std::array<cd, 4> Y() {
  return {cd(0, 0), cd(0, -1), cd(0, 1), cd(0, 0)};
}

std::array<cd, 4> Z() { return {cd(1, 0), cd(0, 0), cd(0, 0), cd(-1, 0)}; }

std::array<cd, 4> H() {
  return {cd(kInvSqrt2, 0), cd(kInvSqrt2, 0), cd(kInvSqrt2, 0),
          cd(-kInvSqrt2, 0)};
}

std::array<cd, 4> Phase(double lambda) {
  return {cd(1, 0), cd(0, 0), cd(0, 0), std::exp(cd(0, lambda))};
}

std::array<cd, 4> S() { return Phase(M_PI / 2.0); }
std::array<cd, 4> T() { return Phase(M_PI / 4.0); }

std::array<cd, 4> RX(double theta) {
  const double c = std::cos(theta / 2.0);
  const double s = std::sin(theta / 2.0);
  return {cd(c, 0), cd(0, -s), cd(0, -s), cd(c, 0)};
}

std::array<cd, 4> RY(double theta) {
  const double c = std::cos(theta / 2.0);
  const double s = std::sin(theta / 2.0);
  return {cd(c, 0), cd(-s, 0), cd(s, 0), cd(c, 0)};
}

std::array<cd, 4> RZ(double theta) {
  return {std::exp(cd(0, -theta / 2.0)), cd(0, 0), cd(0, 0),
          std::exp(cd(0, theta / 2.0))};
}

}  // namespace gates
}  // namespace quasar
