#include "quasar/algorithms.hpp"

#include <algorithm>
#include <cmath>

namespace quasar {

namespace {
std::vector<int> makeRange(int n) {
  std::vector<int> v(n);
  for (int i = 0; i < n; ++i) v[i] = i;
  return v;
}
}  // namespace

// QFT. Textbook construction. With little-endian indexing (qubit 0 = LSB),
// applying H to the most-significant qubit first with a controlled-phase
// ladder, then reversing qubit order with SWAPs, yields the standard DFT:
//   |x> -> (1/sqrt(N)) sum_y exp(2*pi*i*x*y/N) |y>.
Circuit qft(int n) {
  Circuit c(n);
  for (int j = n - 1; j >= 0; --j) {
    c.h(j);
    for (int k = j - 1; k >= 0; --k) {
      const double angle = M_PI / std::pow(2.0, j - k);
      c.controlled("phase", k, j, angle);
    }
  }
  for (int i = 0; i < n / 2; ++i) c.swap(i, n - 1 - i);
  return c;
}

Circuit ghz(int n) {
  Circuit c(n);
  c.h(0);
  for (int i = 0; i < n - 1; ++i) c.cx(i, i + 1);
  return c;
}

Circuit grover(int n, std::uint64_t marked) {
  Circuit c(n);
  const std::size_t N = std::size_t{1} << n;
  const int iters = static_cast<int>(
      std::floor(M_PI / 4.0 * std::sqrt(static_cast<double>(N))));

  for (int q = 0; q < n; ++q) c.h(q);

  // Phase-flip of a chosen basis state via an n-qubit multi-controlled Z (mcz),
  // conjugated by X gates so the flipped state becomes the all-ones state that
  // mcz acts on.
  auto phase_flip_state = [&](std::uint64_t target) {
    for (int q = 0; q < n; ++q)
      if (!((target >> q) & 1ULL)) c.x(q);
    c.add({"mcz", makeRange(n), {}});
    for (int q = 0; q < n; ++q)
      if (!((target >> q) & 1ULL)) c.x(q);
  };

  for (int it = 0; it < iters; ++it) {
    // Oracle.
    phase_flip_state(marked);
    // Diffusion: H^n (2|0><0| - I) H^n. The 2|0><0|-I is -1 times a phase flip
    // of |0>; the global sign is irrelevant.
    for (int q = 0; q < n; ++q) c.h(q);
    phase_flip_state(0);
    for (int q = 0; q < n; ++q) c.h(q);
  }
  return c;
}

Circuit qaoa_ansatz(int n, const std::vector<double>& gammas,
                    const std::vector<double>& betas) {
  Circuit c(n);
  for (int q = 0; q < n; ++q) c.h(q);

  const std::size_t p = std::min(gammas.size(), betas.size());
  for (std::size_t layer = 0; layer < p; ++layer) {
    const double gamma = gammas[layer];
    const double beta = betas[layer];

    // Cost layer: exp(-i gamma sum_{<i,j>} Z_i Z_j) over a ring of n nodes.
    // exp(-i gamma Z_i Z_j) = CX(i,j) RZ(2 gamma)_j CX(i,j).
    const int edges = (n <= 2) ? (n == 2 ? 1 : 0) : n;
    for (int e = 0; e < edges; ++e) {
      const int i = e;
      const int j = (e + 1) % n;
      c.cx(i, j);
      c.rz(j, 2.0 * gamma);
      c.cx(i, j);
    }

    // Mixer: exp(-i beta sum_i X_i) = product RX(2 beta)_i.
    for (int q = 0; q < n; ++q) c.rx(q, 2.0 * beta);
  }
  return c;
}

}  // namespace quasar
