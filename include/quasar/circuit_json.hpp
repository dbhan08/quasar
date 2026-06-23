#pragma once

#include <string>

#include "quasar/circuit.hpp"

namespace quasar {

// Minimal JSON circuit format used by the benchmark harness so that quasar and
// the Python golden reference run the *identical* gate sequence.
//
// Format:
//   {"nqubits": N, "ops": [{"name":"h","qubits":[0],"params":[]}, ...]}
//
// Only the small subset of JSON we emit is parsed (no general JSON support).
Circuit loadCircuitJson(const std::string& text);
Circuit loadCircuitJsonFile(const std::string& path);

}  // namespace quasar
