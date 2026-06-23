#pragma once

#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include "quasar/types.hpp"

namespace quasar {

// Dense state-vector representation of an n-qubit register.
//
// Qubit ordering is little-endian: qubit 0 is the least-significant bit of the
// basis-state index, matching Qiskit. Basis state |b_{n-1} ... b_1 b_0> maps to
// the array index sum_q b_q * 2^q.
class StateVector {
 public:
  explicit StateVector(int nqubits);

  int nqubits() const { return nqubits_; }
  std::size_t dim() const { return amp_.size(); }

  std::vector<cd>& data() { return amp_; }
  const std::vector<cd>& data() const { return amp_; }

  // Reset to |0...0>.
  void reset();

  // Apply a single-qubit 2x2 gate [[a,b],[c,d]] to qubit q.
  void apply1(int q, const std::array<cd, 4>& m);

  // Apply a controlled single-qubit gate: when qubit `control` is 1, apply the
  // 2x2 gate `m` to qubit `target`.
  void applyControlled1(int control, int target, const std::array<cd, 4>& m);

  // Apply a general k-qubit dense gate. `targets` lists the k target qubits
  // (LSB-first within the small unitary's index); `u` is a row-major
  // 2^k x 2^k unitary. The bit position of targets[j] within the gather index
  // is j (i.e. targets[0] is the least-significant gathered bit).
  void applyMulti(const std::vector<int>& targets, const std::vector<cd>& u);

  // Apply a k-qubit DIAGONAL gate. `targets` lists the k target qubits
  // (LSB-first); `diag` is the 2^k diagonal of the unitary, where diag[s]
  // multiplies every amplitude whose target-bit pattern equals s (targets[j] in
  // bit j of s). O(2^n), no gather/scatter, no GEMM. Multi-threaded.
  void applyDiagonal(const std::vector<int>& targets, const std::vector<cd>& diag);

  // Same semantics as applyMulti, but routes the dense application through
  // Apple Accelerate's cblas_zgemm: gather the 2^(n-k) groups as columns of a
  // 2^k x groups matrix, do one complex GEMM (U * cols), then scatter back.
  void applyMultiBlas(const std::vector<int>& targets, const std::vector<cd>& u);

  // Probability of each basis state (|amp|^2).
  std::vector<double> probabilities() const;

  // L2 norm squared (should stay ~1 for unitary evolution).
  double normSquared() const;

  // Sample one basis-state index in the computational basis using `rng`.
  std::uint64_t sample(std::mt19937_64& rng) const;

 private:
  int nqubits_;
  std::vector<cd> amp_;
};

}  // namespace quasar
