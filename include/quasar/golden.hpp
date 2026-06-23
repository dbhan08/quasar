#pragma once

#include <string>
#include <vector>

#include "quasar/types.hpp"

namespace quasar {

struct Golden {
  int nqubits = 0;
  std::string circuit;
  std::string baseline;
  double aer_time_s = 0.0;
  std::vector<cd> state;
};

Golden loadGoldenFile(const std::string& path);

}  // namespace quasar
