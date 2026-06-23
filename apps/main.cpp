#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "quasar/algorithms.hpp"
#include "quasar/circuit.hpp"
#include "quasar/circuit_json.hpp"
#include "quasar/fusion.hpp"
#include "quasar/gates.hpp"
#include "quasar/golden.hpp"
#include "quasar/qasm.hpp"
#include "quasar/statevector.hpp"
#include "quasar/version.hpp"

namespace {

int printTopLevelHelp() {
  std::cout
      << "quasar " << quasar::kVersion
      << " - state-vector quantum circuit simulator (Apple Silicon)\n\n"
      << "Usage:\n"
      << "  quasar --version\n"
      << "  quasar --help\n"
      << "  quasar run <file.qasm> [--shots N]\n"
      << "      Simulate a circuit. Without --shots, prints basis-state\n"
      << "      probabilities; with --shots N, prints sampled measurement "
         "counts.\n"
      << "  quasar bench --circuit <qft|grover|qaoa|ghz> --qubits N "
         "[-n ITERS] [--golden-dir DIR] [--fuse|--blas]\n"
      << "      Run a circuit, compare the state to the golden reference\n"
      << "      (max-abs-diff <= 1e-9), and print a quasar-vs-golden latency "
         "table.\n"
      << "  quasar profile [--qubits N] [--reps R] [--maxk K]\n"
      << "      Sweep fused-block width and report the scalar-vs-zgemm "
         "crossover.\n"
      << "  quasar demo\n"
      << "      Run a 1-qubit Hadamard and print probabilities.\n"
      << "\nQubit ordering is little-endian (qubit 0 = least-significant bit).\n";
  return 0;
}

int benchHelp() {
  std::cout
      << "quasar bench --circuit <qft|grover|qaoa|ghz> --qubits N [-n ITERS] "
         "[--golden-dir DIR]\n\n"
      << "  Loads <DIR>/<circuit>_<N>.circuit.json (the shared IR) and\n"
      << "  <DIR>/<circuit>_<N>.golden.json (the golden statevector + Aer\n"
      << "  timing) produced by bench/golden.py, simulates the circuit in\n"
      << "  quasar, asserts max-abs-diff <= 1e-9 vs golden, and prints a\n"
      << "  latency table (quasar mean over ITERS runs vs the golden Aer time).\n"
      << "  Default DIR is bench_out, default ITERS is 5.\n"
      << "  --fuse applies the gate-fusion pass; --blas additionally routes\n"
      << "  fused dense gates through cblas_zgemm (Accelerate).\n";
  return 0;
}

int cmdBench(int argc, char** argv) {
  std::string circuit;
  int qubits = -1;
  int iters = 5;
  std::string dir = "bench_out";
  bool useFusion = false;
  bool useBlas = false;
  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0)
      return benchHelp();
    else if (std::strcmp(argv[i], "--circuit") == 0 && i + 1 < argc)
      circuit = argv[++i];
    else if (std::strcmp(argv[i], "--qubits") == 0 && i + 1 < argc)
      qubits = std::stoi(argv[++i]);
    else if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc)
      iters = std::stoi(argv[++i]);
    else if (std::strcmp(argv[i], "--golden-dir") == 0 && i + 1 < argc)
      dir = argv[++i];
    else if (std::strcmp(argv[i], "--fuse") == 0)
      useFusion = true;
    else if (std::strcmp(argv[i], "--blas") == 0) {
      useFusion = true;
      useBlas = true;
    } else {
      std::cerr << "quasar bench: unexpected argument '" << argv[i] << "'\n";
      return 1;
    }
  }
  if (circuit.empty() || qubits < 0) {
    std::cerr << "quasar bench: --circuit and --qubits are required\n";
    return benchHelp();
  }

  const std::string base = dir + "/" + circuit + "_" + std::to_string(qubits);
  quasar::Circuit c(0);
  quasar::Golden golden;
  try {
    c = quasar::loadCircuitJsonFile(base + ".circuit.json");
    golden = quasar::loadGoldenFile(base + ".golden.json");
  } catch (const std::exception& e) {
    std::cerr << "quasar bench: " << e.what()
              << "\n  (run: .venv/bin/python bench/golden.py --circuit "
              << circuit << " --qubits " << qubits << " --out " << base
              << ")\n";
    return 1;
  }

  // Optionally precompute the fused circuit (fusion cost is one-time).
  quasar::Circuit fused = useFusion ? quasar::fuse(c, 4) : c;

  // Correctness: one clean run, compare to golden.
  quasar::StateVector sv(c.nqubits());
  if (useFusion)
    quasar::applyFused(fused, sv, useBlas);
  else
    quasar::apply(c, sv);
  double maxdiff = 0.0;
  if (sv.dim() != golden.state.size()) {
    std::cerr << "quasar bench: dimension mismatch with golden\n";
    return 1;
  }
  for (std::size_t i = 0; i < sv.dim(); ++i)
    maxdiff = std::max(maxdiff, std::abs(sv.data()[i] - golden.state[i]));

  // Timing: mean over `iters` runs (fresh state each time).
  double total = 0.0, best = 1e30;
  for (int it = 0; it < iters; ++it) {
    quasar::StateVector s(c.nqubits());
    auto t0 = std::chrono::high_resolution_clock::now();
    if (useFusion)
      quasar::applyFused(fused, s, useBlas);
    else
      quasar::apply(c, s);
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    total += dt;
    best = std::min(best, dt);
  }
  const double mean = total / iters;
  const bool pass = maxdiff <= 1e-9;

  std::cout << "circuit=" << circuit << " qubits=" << qubits
            << " dim=" << sv.dim() << " baseline=" << golden.baseline
            << " mode="
            << (useBlas ? "fused+zgemm" : (useFusion ? "fused" : "unfused"))
            << "\n";
  std::cout << "correctness: max-abs-diff=" << maxdiff << "  "
            << (pass ? "PASS (<=1e-9)" : "FAIL (>1e-9)") << "\n";
  std::cout << "latency (mean over " << iters << " runs):\n";
  std::cout << "  quasar : mean=" << mean * 1e3 << " ms  best=" << best * 1e3
            << " ms\n";
  std::cout << "  golden : " << golden.aer_time_s * 1e3 << " ms ("
            << golden.baseline << ")\n";
  return pass ? 0 : 2;
}

std::string basisLabel(std::size_t idx, int n) {
  std::string s(n, '0');
  for (int q = 0; q < n; ++q)
    if ((idx >> q) & 1ULL) s[n - 1 - q] = '1';  // MSB-left, little-endian
  return s;
}

int runHelp() {
  std::cout << "quasar run <file.qasm> [--shots N]\n\n"
            << "  Parses an OpenQASM 2.0 subset file, simulates it, and prints\n"
            << "  results. Default: probabilities of every basis state with\n"
            << "  probability > 1e-9. With --shots N: sampled counts over N\n"
            << "  measurements.\n";
  return 0;
}

int cmdRun(int argc, char** argv) {
  std::string path;
  long shots = 0;
  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0)
      return runHelp();
    if (std::strcmp(argv[i], "--shots") == 0) {
      if (i + 1 >= argc) {
        std::cerr << "quasar run: --shots requires an argument\n";
        return 1;
      }
      shots = std::stol(argv[++i]);
    } else if (path.empty()) {
      path = argv[i];
    } else {
      std::cerr << "quasar run: unexpected argument '" << argv[i] << "'\n";
      return 1;
    }
  }
  if (path.empty()) {
    std::cerr << "quasar run: missing <file.qasm>\n";
    return runHelp();
  }

  quasar::Circuit c(0);
  try {
    c = quasar::parseQasmFile(path);
  } catch (const std::exception& e) {
    std::cerr << "quasar run: " << e.what() << "\n";
    return 1;
  }

  quasar::StateVector sv(c.nqubits());
  quasar::apply(c, sv);

  if (shots > 0) {
    std::mt19937_64 rng(0xC0FFEE);
    std::unordered_map<std::uint64_t, long> counts;
    for (long s = 0; s < shots; ++s) counts[sv.sample(rng)]++;
    std::cout << "shots=" << shots << "\n";
    for (auto& kv : counts)
      std::cout << basisLabel(kv.first, c.nqubits()) << "  " << kv.second
                << "\n";
  } else {
    auto p = sv.probabilities();
    for (std::size_t i = 0; i < p.size(); ++i)
      if (p[i] > 1e-9)
        std::cout << basisLabel(i, c.nqubits()) << "  " << p[i] << "\n";
  }
  return 0;
}

int profileHelp() {
  std::cout
      << "quasar profile [--qubits N] [--reps R] [--maxk K]\n\n"
      << "  Sweeps fused-block width k = 1..K and, on an N-qubit state, times\n"
      << "  applying one random dense 2^k x 2^k gate via the scalar gather/\n"
      << "  scatter path vs the cblas_zgemm (Accelerate) path. Reports the\n"
      << "  crossover block size where zgemm overtakes the scalar loop.\n"
      << "  Defaults: N=22, R=5, K=8.\n";
  return 0;
}

int cmdProfile(int argc, char** argv) {
  int N = 22;
  int reps = 5;
  int maxk = 8;
  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0)
      return profileHelp();
    else if (std::strcmp(argv[i], "--qubits") == 0 && i + 1 < argc)
      N = std::stoi(argv[++i]);
    else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc)
      reps = std::stoi(argv[++i]);
    else if (std::strcmp(argv[i], "--maxk") == 0 && i + 1 < argc)
      maxk = std::stoi(argv[++i]);
    else {
      std::cerr << "quasar profile: unexpected argument '" << argv[i] << "'\n";
      return 1;
    }
  }
  if (maxk >= N) maxk = N - 1;

  std::mt19937_64 rng(0xBADC0DE);
  std::normal_distribution<double> nd(0.0, 1.0);

  std::cout << "profiler: N=" << N << " qubits, reps=" << reps
            << " (one dense k-qubit gate over all 2^(N-k) groups)\n";
  std::cout << "  k   dim    scalar(ms)   zgemm(ms)   speedup   winner\n";

  int crossover = -1;
  for (int k = 1; k <= maxk; ++k) {
    const std::size_t K = std::size_t{1} << k;
    // Random (non-unitary is fine for timing) dense gate.
    std::vector<quasar::cd> u(K * K);
    for (auto& z : u) z = quasar::cd(nd(rng), nd(rng));
    // Targets = lowest k qubits.
    std::vector<int> targets(k);
    for (int j = 0; j < k; ++j) targets[j] = j;

    // Seed a non-trivial state once.
    quasar::StateVector base(N);
    base.apply1(0, quasar::gates::H());
    base.apply1(N - 1, quasar::gates::H());

    auto timeIt = [&](bool blas) {
      double best = 1e30;
      for (int r = 0; r < reps; ++r) {
        quasar::StateVector s = base;
        auto t0 = std::chrono::high_resolution_clock::now();
        if (blas)
          s.applyMultiBlas(targets, u);
        else
          s.applyMulti(targets, u);
        auto t1 = std::chrono::high_resolution_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
      }
      return best;
    };

    double tscalar = timeIt(false);
    double tblas = timeIt(true);
    double speedup = tscalar / tblas;
    const char* winner = (tblas < tscalar) ? "zgemm" : "scalar";
    if (crossover < 0 && tblas < tscalar) crossover = k;

    std::cout << "  " << k << "   " << K << "\t" << tscalar * 1e3 << "\t"
              << tblas * 1e3 << "\t" << speedup << "x\t" << winner << "\n";
  }

  if (crossover > 0)
    std::cout << "crossover: zgemm overtakes the scalar path at fused block "
                 "size k="
              << crossover << " (dim " << (std::size_t{1} << crossover) << ")\n";
  else
    std::cout << "crossover: zgemm did not overtake the scalar path for k<="
              << maxk << " at N=" << N << "\n";
  return 0;
}

int runDemo() {
  quasar::StateVector sv(1);
  sv.apply1(0, quasar::gates::H());
  auto p = sv.probabilities();
  std::cout << "H|0> probabilities: P(0)=" << p[0] << " P(1)=" << p[1] << "\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return printTopLevelHelp();

  const char* cmd = argv[1];
  if (std::strcmp(cmd, "--version") == 0 || std::strcmp(cmd, "-v") == 0) {
    std::cout << "quasar " << quasar::kVersion << "\n";
    return 0;
  }
  if (std::strcmp(cmd, "--help") == 0 || std::strcmp(cmd, "-h") == 0)
    return printTopLevelHelp();
  if (std::strcmp(cmd, "run") == 0) return cmdRun(argc, argv);
  if (std::strcmp(cmd, "bench") == 0) return cmdBench(argc, argv);
  if (std::strcmp(cmd, "profile") == 0) return cmdProfile(argc, argv);
  if (std::strcmp(cmd, "demo") == 0) return runDemo();

  std::cerr << "quasar: unknown command '" << cmd << "'\n";
  printTopLevelHelp();
  return 1;
}
