#include "quasar/statevector.hpp"

#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <thread>

namespace quasar {

namespace {

// Number of worker partitions for parallel apply loops. One per hardware thread.
unsigned int hwThreads() {
  unsigned int n = std::thread::hardware_concurrency();
  return n == 0 ? 1u : n;
}

// Run `body(p)` for p in [0, parts) across GCD's global concurrent queue when the
// work is large enough to amortize dispatch overhead; otherwise run serially on
// the calling thread. `body` must write disjoint amplitude ranges per p.
template <typename F>
void parallelFor(std::size_t parts, std::size_t totalWork, F&& body) {
  // Below this much work, threading overhead dominates; stay serial.
  constexpr std::size_t kParallelThreshold = 1u << 14;
  if (parts <= 1 || totalWork < kParallelThreshold) {
    for (std::size_t p = 0; p < parts; ++p) body(p);
    return;
  }
  dispatch_queue_t q =
      dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
  dispatch_apply(parts, q, ^(std::size_t p) { body(p); });
}

}  // namespace

StateVector::StateVector(int nqubits) : nqubits_(nqubits) {
  if (nqubits < 0 || nqubits > 30) {
    throw std::invalid_argument("nqubits out of supported range [0, 30]");
  }
  amp_.assign(std::size_t{1} << nqubits, cd(0.0, 0.0));
  amp_[0] = cd(1.0, 0.0);
}

void StateVector::reset() {
  std::fill(amp_.begin(), amp_.end(), cd(0.0, 0.0));
  amp_[0] = cd(1.0, 0.0);
}

void StateVector::apply1(int q, const std::array<cd, 4>& m) {
  if (q < 0 || q >= nqubits_) throw std::out_of_range("qubit index out of range");
  const std::size_t bit = std::size_t{1} << q;
  const std::size_t n = amp_.size();
  const cd a = m[0], b = m[1], c = m[2], d = m[3];
  cd* amp = amp_.data();
  // Iterate over the n/2 amplitude pairs. Pair p maps to a low index i with bit
  // q cleared: i = (p with the low q bits kept) | ((p's high bits) << (q+1)).
  const std::size_t low_mask = bit - 1;            // bits [0, q)
  const std::size_t npairs = n >> 1;               // n/2
  const unsigned int parts = hwThreads();
  const std::size_t chunk = (npairs + parts - 1) / parts;
  parallelFor(parts, n, [&](std::size_t pt) {
    const std::size_t begin = pt * chunk;
    if (begin >= npairs) return;
    const std::size_t end = std::min(begin + chunk, npairs);
    for (std::size_t p = begin; p < end; ++p) {
      const std::size_t i = (p & low_mask) | ((p & ~low_mask) << 1);
      const std::size_t j = i | bit;
      const cd x0 = amp[i];
      const cd x1 = amp[j];
      amp[i] = a * x0 + b * x1;
      amp[j] = c * x0 + d * x1;
    }
  });
}

void StateVector::applyControlled1(int control, int target,
                                   const std::array<cd, 4>& m) {
  if (control < 0 || control >= nqubits_ || target < 0 || target >= nqubits_)
    throw std::out_of_range("qubit index out of range");
  if (control == target) throw std::invalid_argument("control == target");
  const std::size_t cbit = std::size_t{1} << control;
  const std::size_t tbit = std::size_t{1} << target;
  const std::size_t n = amp_.size();
  const cd a = m[0], b = m[1], c = m[2], d = m[3];
  for (std::size_t i = 0; i < n; ++i) {
    if (i & tbit) continue;          // process target-bit-0 elements only
    if (!(i & cbit)) continue;       // control must be 1
    const std::size_t j = i | tbit;
    const cd x0 = amp_[i];
    const cd x1 = amp_[j];
    amp_[i] = a * x0 + b * x1;
    amp_[j] = c * x0 + d * x1;
  }
}

void StateVector::applyMulti(const std::vector<int>& targets,
                             const std::vector<cd>& u) {
  const int k = static_cast<int>(targets.size());
  if (k == 0) return;
  const std::size_t dimk = std::size_t{1} << k;
  if (u.size() != dimk * dimk)
    throw std::invalid_argument("unitary size mismatch with target count");
  for (int t : targets) {
    if (t < 0 || t >= nqubits_) throw std::out_of_range("qubit index out of range");
  }

  // Mask of all target bits, to iterate over "other" qubit settings.
  std::size_t target_mask = 0;
  std::vector<std::size_t> tbits(k);
  for (int j = 0; j < k; ++j) {
    tbits[j] = std::size_t{1} << targets[j];
    target_mask |= tbits[j];
  }

  const int n_other = nqubits_ - k;
  const std::size_t n_groups = std::size_t{1} << n_other;

  // Enumerate the "other" qubit indices (bits not in target_mask).
  std::vector<std::size_t> other_bits;
  other_bits.reserve(n_other);
  for (int q = 0; q < nqubits_; ++q) {
    std::size_t b = std::size_t{1} << q;
    if (!(b & target_mask)) other_bits.push_back(b);
  }

  // Precompute, for each target sub-index s, the bit-mask it contributes.
  std::vector<std::size_t> smask(dimk);
  for (std::size_t s = 0; s < dimk; ++s) {
    std::size_t mm = 0;
    for (int j = 0; j < k; ++j)
      if (s & (std::size_t{1} << j)) mm |= tbits[j];
    smask[s] = mm;
  }

  cd* amp = amp_.data();
  const cd* uptr = u.data();
  const unsigned int parts = hwThreads();
  const std::size_t chunk = (n_groups + parts - 1) / parts;
  parallelFor(parts, amp_.size(), [&](std::size_t pt) {
    const std::size_t gbegin = pt * chunk;
    if (gbegin >= n_groups) return;
    const std::size_t gend = std::min(gbegin + chunk, n_groups);
    std::vector<cd> in(dimk), out(dimk);
    for (std::size_t g = gbegin; g < gend; ++g) {
      // Build the base index for this group from the "other" bits.
      std::size_t base = 0;
      for (int ob = 0; ob < n_other; ++ob)
        if (g & (std::size_t{1} << ob)) base |= other_bits[ob];
      // Gather the 2^k amplitudes addressed by target-bit patterns.
      for (std::size_t s = 0; s < dimk; ++s) in[s] = amp[base | smask[s]];
      // out = U * in
      for (std::size_t r = 0; r < dimk; ++r) {
        cd acc(0.0, 0.0);
        const cd* urow = &uptr[r * dimk];
        for (std::size_t cc = 0; cc < dimk; ++cc) acc += urow[cc] * in[cc];
        out[r] = acc;
      }
      // Scatter back.
      for (std::size_t s = 0; s < dimk; ++s) amp[base | smask[s]] = out[s];
    }
  });
}

void StateVector::applyDiagonal(const std::vector<int>& targets,
                                const std::vector<cd>& diag) {
  const int k = static_cast<int>(targets.size());
  if (k == 0) return;
  const std::size_t dimk = std::size_t{1} << k;
  if (diag.size() != dimk)
    throw std::invalid_argument("diagonal size mismatch with target count");
  for (int t : targets) {
    if (t < 0 || t >= nqubits_) throw std::out_of_range("qubit index out of range");
  }

  std::vector<std::size_t> tbits(k);
  for (int j = 0; j < k; ++j) tbits[j] = std::size_t{1} << targets[j];

  cd* amp = amp_.data();
  const cd* dptr = diag.data();
  const std::size_t n = amp_.size();
  const unsigned int parts = hwThreads();
  const std::size_t chunk = (n + parts - 1) / parts;
  parallelFor(parts, n, [&](std::size_t pt) {
    const std::size_t begin = pt * chunk;
    if (begin >= n) return;
    const std::size_t end = std::min(begin + chunk, n);
    for (std::size_t i = begin; i < end; ++i) {
      // Select the diagonal entry by the target-qubit bit pattern of index i.
      std::size_t s = 0;
      for (int j = 0; j < k; ++j)
        if (i & tbits[j]) s |= (std::size_t{1} << j);
      amp[i] *= dptr[s];
    }
  });
}

void StateVector::applyMultiBlas(const std::vector<int>& targets,
                                 const std::vector<cd>& u) {
  const int k = static_cast<int>(targets.size());
  if (k == 0) return;
  const std::size_t K = std::size_t{1} << k;  // rows/cols of the unitary
  if (u.size() != K * K)
    throw std::invalid_argument("unitary size mismatch with target count");
  for (int t : targets) {
    if (t < 0 || t >= nqubits_) throw std::out_of_range("qubit index out of range");
  }

  std::size_t target_mask = 0;
  std::vector<std::size_t> tbits(k);
  for (int j = 0; j < k; ++j) {
    tbits[j] = std::size_t{1} << targets[j];
    target_mask |= tbits[j];
  }
  const int n_other = nqubits_ - k;
  const std::size_t G = std::size_t{1} << n_other;  // number of groups (columns)

  std::vector<std::size_t> other_bits;
  other_bits.reserve(n_other);
  for (int q = 0; q < nqubits_; ++q) {
    std::size_t b = std::size_t{1} << q;
    if (!(b & target_mask)) other_bits.push_back(b);
  }

  // Precompute, for each group, its base index (the spectator-bit setting).
  std::vector<std::size_t> base(G);
  for (std::size_t g = 0; g < G; ++g) {
    std::size_t bidx = 0;
    for (int ob = 0; ob < n_other; ++ob)
      if (g & (std::size_t{1} << ob)) bidx |= other_bits[ob];
    base[g] = bidx;
  }
  // Precompute, for each target sub-index s, its contribution mask.
  std::vector<std::size_t> smask(K);
  for (std::size_t s = 0; s < K; ++s) {
    std::size_t m = 0;
    for (int j = 0; j < k; ++j)
      if (s & (std::size_t{1} << j)) m |= tbits[j];
    smask[s] = m;
  }

  // Build U in column-major order (Accelerate/BLAS is column-major). u is stored
  // row-major as u[r*K + c]; column-major element (r,c) lives at c*K + r.
  std::vector<cd> Ucm(K * K);
  for (std::size_t r = 0; r < K; ++r)
    for (std::size_t c = 0; c < K; ++c) Ucm[c * K + r] = u[r * K + c];

  // Gather X (K x G), column-major: column g, row s = amp at base[g] | smask[s].
  std::vector<cd> X(K * G);
  {
    cd* amp = amp_.data();
    const unsigned int parts = hwThreads();
    const std::size_t chunk = (G + parts - 1) / parts;
    parallelFor(parts, amp_.size(), [&](std::size_t pt) {
      const std::size_t gb = pt * chunk;
      if (gb >= G) return;
      const std::size_t ge = std::min(gb + chunk, G);
      for (std::size_t g = gb; g < ge; ++g) {
        const std::size_t b = base[g];
        cd* col = &X[g * K];
        for (std::size_t s = 0; s < K; ++s) col[s] = amp[b | smask[s]];
      }
    });
  }

  // Y = U * X, both column-major. zgemm: C(MxN) = alpha*A(MxK)*B(KxN)+beta*C.
  std::vector<cd> Y(K * G);
  const cd alpha(1.0, 0.0), beta(0.0, 0.0);
  cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
              static_cast<int>(K), static_cast<int>(G), static_cast<int>(K),
              &alpha, Ucm.data(), static_cast<int>(K), X.data(),
              static_cast<int>(K), &beta, Y.data(), static_cast<int>(K));

  // Scatter back.
  {
    cd* amp = amp_.data();
    const unsigned int parts = hwThreads();
    const std::size_t chunk = (G + parts - 1) / parts;
    parallelFor(parts, amp_.size(), [&](std::size_t pt) {
      const std::size_t gb = pt * chunk;
      if (gb >= G) return;
      const std::size_t ge = std::min(gb + chunk, G);
      for (std::size_t g = gb; g < ge; ++g) {
        const std::size_t b = base[g];
        const cd* col = &Y[g * K];
        for (std::size_t s = 0; s < K; ++s) amp[b | smask[s]] = col[s];
      }
    });
  }
}

std::vector<double> StateVector::probabilities() const {
  std::vector<double> p(amp_.size());
  for (std::size_t i = 0; i < amp_.size(); ++i) p[i] = std::norm(amp_[i]);
  return p;
}

double StateVector::normSquared() const {
  double s = 0.0;
  for (const cd& a : amp_) s += std::norm(a);
  return s;
}

std::uint64_t StateVector::sample(std::mt19937_64& rng) const {
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  const double r = dist(rng);
  double cum = 0.0;
  for (std::size_t i = 0; i < amp_.size(); ++i) {
    cum += std::norm(amp_[i]);
    if (r < cum) return static_cast<std::uint64_t>(i);
  }
  return static_cast<std::uint64_t>(amp_.size() - 1);
}

}  // namespace quasar
