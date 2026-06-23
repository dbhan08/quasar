#include "quasar/fusion.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "quasar/gates.hpp"

namespace quasar {

namespace {

// Build the 4x4 controlled-U matrix for control=qubit-bit-1, target=qubit-bit-0
// under applyMulti convention with targets ordered {target, control} so that
// bit0 = target, bit1 = control. We will instead always emit dense gates with a
// canonical target ordering = sorted qubit indices; callers below handle the
// bit mapping explicitly. To keep things simple and robust, denseFor builds the
// matrix directly over its declared `targets` ordering.

// Helpers to construct a controlled single-qubit gate over targets = {control,
// target} where targets[0] is bit0. We choose targets = {target, control}:
//   bit0 = target, bit1 = control.
// Matrix is 4x4: when control bit==1, apply 2x2 m to target bit; else identity.
std::vector<cd> controlled4(const std::array<cd, 4>& m) {
  // index s: bit0=target(t), bit1=control(c). s = t + 2c.
  // When c==1: apply m to t. When c==0: identity.
  std::vector<cd> u(16, cd(0, 0));
  auto at = [&](int r, int col) -> cd& { return u[r * 4 + col]; };
  // c==0 block: states 0 (t=0,c=0) and 1 (t=1,c=0) unchanged.
  at(0, 0) = cd(1, 0);
  at(1, 1) = cd(1, 0);
  // c==1 block: states 2 (t=0,c=1) and 3 (t=1,c=1) get m on the t bit.
  // m = [[a,b],[c,d]] maps |t=0>->a|0>+c|1>, |t=1>->b|0>+d|1>.
  const cd a = m[0], b = m[1], cc = m[2], d = m[3];
  at(2, 2) = a;  // out t=0,c=1 from in t=0,c=1
  at(3, 2) = cc;  // out t=1 from in t=0
  at(2, 3) = b;  // out t=0 from in t=1
  at(3, 3) = d;  // out t=1 from in t=1
  return u;
}

std::vector<cd> swap4() {
  // targets {a,b}: bit0=a, bit1=b. SWAP exchanges the two bits.
  // s = ba + 2*bb. Map (ba,bb)->(bb,ba).
  std::vector<cd> u(16, cd(0, 0));
  auto at = [&](int r, int col) -> cd& { return u[r * 4 + col]; };
  for (int ba = 0; ba < 2; ++ba)
    for (int bb = 0; bb < 2; ++bb) {
      int in = ba + 2 * bb;
      int out = bb + 2 * ba;
      at(out, in) = cd(1, 0);
    }
  return u;
}

std::vector<cd> mat1(const std::array<cd, 4>& m) {
  return {m[0], m[1], m[2], m[3]};
}

}  // namespace

DenseGate denseFor(const GateOp& op) {
  const std::string& n = op.name;
  DenseGate g;

  auto single = [&](const std::array<cd, 4>& m) {
    g.targets = {op.qubits[0]};
    g.u = mat1(m);
  };

  if (n == "h") return (single(gates::H()), g);
  if (n == "x") return (single(gates::X()), g);
  if (n == "y") return (single(gates::Y()), g);
  if (n == "z") return (single(gates::Z()), g);
  if (n == "s") return (single(gates::S()), g);
  if (n == "t") return (single(gates::T()), g);
  if (n == "rx") return (single(gates::RX(op.params.at(0))), g);
  if (n == "ry") return (single(gates::RY(op.params.at(0))), g);
  if (n == "rz") return (single(gates::RZ(op.params.at(0))), g);
  if (n == "phase") return (single(gates::Phase(op.params.at(0))), g);

  if (n == "cx" || n == "cz" || (n.size() > 1 && n[0] == 'c' && n != "cx" &&
                                 n != "cz")) {
    // Controlled single-qubit gate. qubits[0]=control, qubits[1]=target.
    std::array<cd, 4> m;
    if (n == "cx")
      m = gates::X();
    else if (n == "cz")
      m = gates::Z();
    else {
      const std::string base = n.substr(1);
      if (base == "x") m = gates::X();
      else if (base == "y") m = gates::Y();
      else if (base == "z") m = gates::Z();
      else if (base == "h") m = gates::H();
      else if (base == "s") m = gates::S();
      else if (base == "t") m = gates::T();
      else if (base == "rx") m = gates::RX(op.params.at(0));
      else if (base == "ry") m = gates::RY(op.params.at(0));
      else if (base == "rz") m = gates::RZ(op.params.at(0));
      else if (base == "phase") m = gates::Phase(op.params.at(0));
      else throw std::invalid_argument("denseFor: unknown controlled gate " + n);
    }
    // targets = {target, control}: bit0=target, bit1=control.
    g.targets = {op.qubits[1], op.qubits[0]};
    g.u = controlled4(m);
    return g;
  }

  if (n == "swap") {
    g.targets = {op.qubits[0], op.qubits[1]};
    g.u = swap4();
    return g;
  }

  if (n == "mcz") {
    const int k = static_cast<int>(op.qubits.size());
    const std::size_t dim = std::size_t{1} << k;
    g.targets = op.qubits;
    g.u.assign(dim * dim, cd(0, 0));
    for (std::size_t i = 0; i < dim; ++i) g.u[i * dim + i] = cd(1, 0);
    // Flip the all-ones diagonal entry.
    g.u[(dim - 1) * dim + (dim - 1)] = cd(-1, 0);
    return g;
  }

  throw std::invalid_argument("denseFor: unsupported gate " + n);
}

namespace {

// Expand a dense gate (targets T, matrix U) onto a merged sorted qubit set M
// (with |M| = w). Returns a 2^w x 2^w matrix. T must be a subset of M.
// Bit j of the merged index corresponds to merged[j].
std::vector<cd> expandTo(const DenseGate& gate, const std::vector<int>& merged) {
  const int w = static_cast<int>(merged.size());
  const std::size_t W = std::size_t{1} << w;
  const int k = static_cast<int>(gate.targets.size());
  const std::size_t K = std::size_t{1} << k;

  // Map each target qubit to its bit position in the merged index.
  std::vector<int> tpos(k);
  for (int j = 0; j < k; ++j) {
    auto it = std::find(merged.begin(), merged.end(), gate.targets[j]);
    tpos[j] = static_cast<int>(it - merged.begin());
  }
  // Spectator bit positions (merged bits not in targets).
  std::vector<int> spec;
  for (int p = 0; p < w; ++p) {
    bool isTarget = false;
    for (int j = 0; j < k; ++j)
      if (tpos[j] == p) { isTarget = true; break; }
    if (!isTarget) spec.push_back(p);
  }

  std::vector<cd> out(W * W, cd(0, 0));
  // For each merged input index, extract target sub-index (in) and spectator
  // bits; the output ranges over target sub-index (row) with the same
  // spectator bits.
  for (std::size_t in = 0; in < W; ++in) {
    // Gate input sub-index from target bits of `in`.
    std::size_t gin = 0;
    for (int j = 0; j < k; ++j)
      if (in & (std::size_t{1} << tpos[j])) gin |= (std::size_t{1} << j);
    // Spectator portion (kept fixed).
    std::size_t specbits = in;
    for (int j = 0; j < k; ++j) specbits &= ~(std::size_t{1} << tpos[j]);

    for (std::size_t grow = 0; grow < K; ++grow) {
      cd val = gate.u[grow * K + gin];
      if (val == cd(0, 0)) continue;
      // Build merged output index: spectator bits + target bits from grow.
      std::size_t outIdx = specbits;
      for (int j = 0; j < k; ++j)
        if (grow & (std::size_t{1} << j)) outIdx |= (std::size_t{1} << tpos[j]);
      out[outIdx * W + in] = val;
    }
  }
  return out;
}

// Matrix multiply C = A * B for square WxW row-major matrices.
std::vector<cd> matmul(const std::vector<cd>& A, const std::vector<cd>& B,
                       std::size_t W) {
  std::vector<cd> C(W * W, cd(0, 0));
  for (std::size_t i = 0; i < W; ++i)
    for (std::size_t kk = 0; kk < W; ++kk) {
      cd a = A[i * W + kk];
      if (a == cd(0, 0)) continue;
      for (std::size_t j = 0; j < W; ++j) C[i * W + j] += a * B[kk * W + j];
    }
  return C;
}

// Number of distinct qubits in the union of a set and a gate's qubits.
std::vector<int> unionQubits(const std::vector<int>& base,
                             const std::vector<int>& add) {
  std::vector<int> u = base;
  for (int q : add)
    if (std::find(u.begin(), u.end(), q) == u.end()) u.push_back(q);
  std::sort(u.begin(), u.end());
  return u;
}

}  // namespace

Circuit fuse(const Circuit& c, int maxWidth) {
  Circuit out(c.nqubits());
  const auto& ops = c.ops();
  std::size_t i = 0;
  while (i < ops.size()) {
    // Start a new fusion block with op i.
    DenseGate first = denseFor(ops[i]);
    std::vector<int> merged = first.targets;
    std::sort(merged.begin(), merged.end());
    // Collect ops indices in the block.
    std::vector<std::size_t> block = {i};
    std::size_t j = i + 1;
    while (j < ops.size()) {
      DenseGate cand = denseFor(ops[j]);
      std::vector<int> cand_union = unionQubits(merged, cand.targets);
      if (static_cast<int>(cand_union.size()) > maxWidth) break;
      merged = cand_union;
      block.push_back(j);
      ++j;
    }

    if (block.size() == 1) {
      // Single gate fused trivially: still emit as a fused dense op so the
      // apply path is uniform, expanded over its own (sorted) targets.
      std::sort(merged.begin(), merged.end());
    }

    // Build the fused unitary over `merged` by composing gates in order.
    const int w = static_cast<int>(merged.size());
    const std::size_t W = std::size_t{1} << w;
    // Start with identity.
    std::vector<cd> acc(W * W, cd(0, 0));
    for (std::size_t d = 0; d < W; ++d) acc[d * W + d] = cd(1, 0);
    for (std::size_t bi : block) {
      DenseGate dg = denseFor(ops[bi]);
      std::vector<cd> e = expandTo(dg, merged);
      acc = matmul(e, acc, W);  // apply e after acc: e * acc
    }

    // Detect whether the composed block is DIAGONAL: every off-diagonal entry
    // is negligible. If so, emit a compact "fused_diag" op carrying only the
    // 2^w diagonal entries, routed through the cheap O(2^n) diagonal apply.
    constexpr double kDiagTol = 1e-12;
    bool isDiag = true;
    for (std::size_t r = 0; r < W && isDiag; ++r)
      for (std::size_t cc = 0; cc < W; ++cc) {
        if (r == cc) continue;
        if (std::abs(acc[r * W + cc]) > kDiagTol) { isDiag = false; break; }
      }

    GateOp fop;
    fop.qubits = merged;
    if (isDiag) {
      fop.name = "fused_diag";
      fop.params.reserve(2 * W);
      for (std::size_t d = 0; d < W; ++d) {
        fop.params.push_back(acc[d * W + d].real());
        fop.params.push_back(acc[d * W + d].imag());
      }
    } else {
      // Pack the flattened unitary into params as [re,im,...].
      fop.name = "fused";
      fop.params.reserve(2 * W * W);
      for (std::size_t idx = 0; idx < W * W; ++idx) {
        fop.params.push_back(acc[idx].real());
        fop.params.push_back(acc[idx].imag());
      }
    }
    out.add(std::move(fop));
    i = j;
  }
  return out;
}

void applyFused(const Circuit& c, StateVector& sv, bool useBlas) {
  for (const GateOp& op : c.ops()) {
    if (op.name == "fused_diag") {
      // Diagonal block: O(2^n) elementwise scaling, regardless of useBlas.
      const int w = static_cast<int>(op.qubits.size());
      const std::size_t W = std::size_t{1} << w;
      std::vector<cd> diag(W);
      for (std::size_t idx = 0; idx < W; ++idx)
        diag[idx] = cd(op.params[2 * idx], op.params[2 * idx + 1]);
      sv.applyDiagonal(op.qubits, diag);
    } else if (op.name == "fused") {
      const int w = static_cast<int>(op.qubits.size());
      const std::size_t W = std::size_t{1} << w;
      std::vector<cd> u(W * W);
      for (std::size_t idx = 0; idx < W * W; ++idx)
        u[idx] = cd(op.params[2 * idx], op.params[2 * idx + 1]);
      if (useBlas)
        sv.applyMultiBlas(op.qubits, u);
      else
        sv.applyMulti(op.qubits, u);
    } else {
      // Fall back to the standard single-op apply for any non-fused op.
      Circuit one(c.nqubits());
      one.add(op);
      apply(one, sv);
    }
  }
}

}  // namespace quasar
