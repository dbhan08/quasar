#pragma once

#include <string>
#include <vector>

#include "quasar/statevector.hpp"

namespace quasar {

// A single operation in the circuit IR.
//   name   : gate name, e.g. "h", "cx", "rz".
//   qubits : operand qubits. For controlled gates, control(s) first, target
//            last, matching the builder methods below.
//   params : rotation/phase angles (radians); empty for non-parametric gates.
struct GateOp {
  std::string name;
  std::vector<int> qubits;
  std::vector<double> params;
};

// Ordered list of gate operations on an n-qubit register, plus a fluent builder.
class Circuit {
 public:
  explicit Circuit(int nqubits) : nqubits_(nqubits) {}

  int nqubits() const { return nqubits_; }
  const std::vector<GateOp>& ops() const { return ops_; }
  std::vector<GateOp>& ops() { return ops_; }

  // Single-qubit gates.
  Circuit& h(int q);
  Circuit& x(int q);
  Circuit& y(int q);
  Circuit& z(int q);
  Circuit& s(int q);
  Circuit& t(int q);
  Circuit& rx(int q, double theta);
  Circuit& ry(int q, double theta);
  Circuit& rz(int q, double theta);
  Circuit& phase(int q, double lambda);

  // Two-qubit gates.
  Circuit& cx(int control, int target);
  Circuit& cz(int control, int target);
  Circuit& swap(int a, int b);

  // Generic controlled single-qubit gate by name ("x","y","z","h","s","t") or a
  // controlled rotation ("rx","ry","rz","phase" with one param).
  Circuit& controlled(const std::string& gate, int control, int target,
                      double param = 0.0);

  // Append a raw op.
  Circuit& add(GateOp op);

 private:
  int nqubits_;
  std::vector<GateOp> ops_;
};

// Apply an entire circuit to a state vector (mutating it in place).
void apply(const Circuit& c, StateVector& sv);

}  // namespace quasar
