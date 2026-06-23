#pragma once

#include <vector>

#include "quasar/circuit.hpp"
#include "quasar/types.hpp"

namespace quasar {

// Return the dense unitary for a single GateOp, along with the target qubit
// list it acts on. The unitary is row-major 2^k x 2^k, where k = targets.size()
// and target[j] occupies bit j of the gathered index (LSB-first), matching
// StateVector::applyMulti's convention.
struct DenseGate {
  std::vector<int> targets;
  std::vector<cd> u;  // (2^k * 2^k), row-major
};

DenseGate denseFor(const GateOp& op);

// Greedily fuse consecutive gates whose combined qubit-set width <= maxWidth
// into single fused dense gates. Returns a new circuit whose ops are all "fused"
// gates (name == "fused", params carry the flattened unitary, qubits carry the
// targets). The fused circuit is state-for-state identical to the input.
Circuit fuse(const Circuit& c, int maxWidth = 4);

// Apply a circuit that may contain "fused" ops (as produced by fuse()) plus
// ordinary ops, to a state vector. When `useBlas` is true, fused dense gates
// route through cblas_zgemm (Accelerate); otherwise through the scalar
// gather/scatter path.
void applyFused(const Circuit& c, StateVector& sv, bool useBlas = false);

}  // namespace quasar
