#pragma once

#include <vector>

#include "quasar/circuit.hpp"

namespace quasar {

// Quantum Fourier Transform on n qubits.
// Uses the standard H + controlled-phase ladder, with final SWAPs so the output
// ordering matches the textbook/analytic DFT under little-endian indexing.
Circuit qft(int n);

// GHZ state preparation: (|0...0> + |1...1>)/sqrt2.
Circuit ghz(int n);

// Grover search over n qubits for a single marked basis state `marked`.
// Runs the optimal ~floor(pi/4 * sqrt(2^n)) iterations.
Circuit grover(int n, std::uint64_t marked);

// QAOA ansatz for a ring-Ising cost Hamiltonian H_C = sum_i Z_i Z_{i+1}
// (indices mod n). One layer per (gamma, beta) pair. Starts from the uniform
// superposition (H on all qubits).
Circuit qaoa_ansatz(int n, const std::vector<double>& gammas,
                    const std::vector<double>& betas);

}  // namespace quasar
