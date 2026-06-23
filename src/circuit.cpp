#include "quasar/circuit.hpp"

#include <array>
#include <stdexcept>

#include "quasar/gates.hpp"

namespace quasar {

Circuit& Circuit::add(GateOp op) {
  ops_.push_back(std::move(op));
  return *this;
}

Circuit& Circuit::h(int q) { return add({"h", {q}, {}}); }
Circuit& Circuit::x(int q) { return add({"x", {q}, {}}); }
Circuit& Circuit::y(int q) { return add({"y", {q}, {}}); }
Circuit& Circuit::z(int q) { return add({"z", {q}, {}}); }
Circuit& Circuit::s(int q) { return add({"s", {q}, {}}); }
Circuit& Circuit::t(int q) { return add({"t", {q}, {}}); }
Circuit& Circuit::rx(int q, double theta) { return add({"rx", {q}, {theta}}); }
Circuit& Circuit::ry(int q, double theta) { return add({"ry", {q}, {theta}}); }
Circuit& Circuit::rz(int q, double theta) { return add({"rz", {q}, {theta}}); }
Circuit& Circuit::phase(int q, double lambda) {
  return add({"phase", {q}, {lambda}});
}

Circuit& Circuit::cx(int control, int target) {
  return add({"cx", {control, target}, {}});
}
Circuit& Circuit::cz(int control, int target) {
  return add({"cz", {control, target}, {}});
}
Circuit& Circuit::swap(int a, int b) { return add({"swap", {a, b}, {}}); }

Circuit& Circuit::controlled(const std::string& gate, int control, int target,
                            double param) {
  return add({"c" + gate, {control, target}, {param}});
}

namespace {

// Resolve a single-qubit gate name (+ optional param) to its 2x2 matrix.
std::array<cd, 4> resolve1(const std::string& name,
                           const std::vector<double>& params) {
  if (name == "h") return gates::H();
  if (name == "x") return gates::X();
  if (name == "y") return gates::Y();
  if (name == "z") return gates::Z();
  if (name == "s") return gates::S();
  if (name == "t") return gates::T();
  if (name == "rx") return gates::RX(params.at(0));
  if (name == "ry") return gates::RY(params.at(0));
  if (name == "rz") return gates::RZ(params.at(0));
  if (name == "phase") return gates::Phase(params.at(0));
  throw std::invalid_argument("unknown single-qubit gate: " + name);
}

}  // namespace

void apply(const Circuit& c, StateVector& sv) {
  for (const GateOp& op : c.ops()) {
    const std::string& n = op.name;
    if (n == "swap") {
      // SWAP = CX(a,b) CX(b,a) CX(a,b).
      const int a = op.qubits[0], b = op.qubits[1];
      sv.applyControlled1(a, b, gates::X());
      sv.applyControlled1(b, a, gates::X());
      sv.applyControlled1(a, b, gates::X());
    } else if (n == "cx") {
      sv.applyControlled1(op.qubits[0], op.qubits[1], gates::X());
    } else if (n == "cz") {
      sv.applyControlled1(op.qubits[0], op.qubits[1], gates::Z());
    } else if (n == "mcz") {
      // Multi-controlled Z: flip the sign of every amplitude whose listed
      // qubits are all 1. Diagonal, so we can sweep the array directly.
      std::size_t mask = 0;
      for (int q : op.qubits) mask |= std::size_t{1} << q;
      auto& data = sv.data();
      for (std::size_t i = 0; i < data.size(); ++i)
        if ((i & mask) == mask) data[i] = -data[i];
    } else if (n.size() > 1 && n[0] == 'c') {
      // Controlled single-qubit gate, e.g. "crz", "ch", "cphase".
      const std::string base = n.substr(1);
      sv.applyControlled1(op.qubits[0], op.qubits[1],
                          resolve1(base, op.params));
    } else {
      sv.apply1(op.qubits[0], resolve1(n, op.params));
    }
  }
}

}  // namespace quasar
