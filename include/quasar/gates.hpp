#pragma once

#include <array>

#include "quasar/types.hpp"

namespace quasar {

// Single-qubit 2x2 gates, stored row-major as [a, b, c, d] = [[a,b],[c,d]].
namespace gates {

std::array<cd, 4> X();
std::array<cd, 4> Y();
std::array<cd, 4> Z();
std::array<cd, 4> H();
std::array<cd, 4> S();
std::array<cd, 4> T();

std::array<cd, 4> RX(double theta);
std::array<cd, 4> RY(double theta);
std::array<cd, 4> RZ(double theta);
std::array<cd, 4> Phase(double lambda);

}  // namespace gates
}  // namespace quasar
