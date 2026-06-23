#pragma once

#include <string>

#include "quasar/circuit.hpp"

namespace quasar {

// Parse an OpenQASM 2.0 subset into a Circuit.
//
// Supported:
//   OPENQASM 2.0;
//   include "qelib1.inc";            (ignored)
//   qreg q[n];                       (defines register size; one register)
//   h|x|y|z|s|t q[i];
//   rx|ry|rz(theta) q[i];
//   cx|cz|swap q[i], q[j];
//   // line comments
//
// Throws std::runtime_error on malformed input.
Circuit parseQasm(const std::string& text);

// Convenience: read a file and parse it.
Circuit parseQasmFile(const std::string& path);

}  // namespace quasar
