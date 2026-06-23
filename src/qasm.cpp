#include "quasar/qasm.hpp"

#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace quasar {

namespace {

// Strip whitespace from both ends.
std::string trim(const std::string& s) {
  std::size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

// Evaluate a minimal arithmetic expression supporting pi, *, /, +, -, and
// numeric literals. Enough for QASM rotation angles like "pi/2", "3*pi/4".
double evalAngle(const std::string& expr) {
  // Tokenize into numbers and operators, then a simple left-to-right pass with
  // */ before +-. Small hand-rolled evaluator.
  std::string e;
  for (char ch : expr)
    if (!std::isspace(static_cast<unsigned char>(ch))) e += ch;

  std::vector<double> nums;
  std::vector<char> ops;
  std::size_t i = 0;
  auto readNumber = [&](void) -> double {
    if (i < e.size() && (e[i] == 'p') && i + 1 < e.size() && e[i + 1] == 'i') {
      i += 2;
      return M_PI;
    }
    std::size_t start = i;
    while (i < e.size() &&
           (std::isdigit(static_cast<unsigned char>(e[i])) || e[i] == '.' ||
            e[i] == 'e' || e[i] == 'E' ||
            ((e[i] == '+' || e[i] == '-') && i > start &&
             (e[i - 1] == 'e' || e[i - 1] == 'E')))) {
      ++i;
    }
    if (start == i)
      throw std::runtime_error("qasm: bad angle expression: " + expr);
    return std::stod(e.substr(start, i - start));
  };

  // Optional leading sign.
  double sign = 1.0;
  if (i < e.size() && (e[i] == '+' || e[i] == '-')) {
    if (e[i] == '-') sign = -1.0;
    ++i;
  }
  nums.push_back(sign * readNumber());
  while (i < e.size()) {
    char op = e[i++];
    if (op != '+' && op != '-' && op != '*' && op != '/')
      throw std::runtime_error("qasm: bad operator in angle: " + expr);
    double sgn = 1.0;
    if (i < e.size() && (e[i] == '+' || e[i] == '-')) {
      if (e[i] == '-') sgn = -1.0;
      ++i;
    }
    nums.push_back(sgn * readNumber());
    ops.push_back(op);
  }

  // First pass: * and /.
  for (std::size_t k = 0; k < ops.size();) {
    if (ops[k] == '*' || ops[k] == '/') {
      double r = (ops[k] == '*') ? nums[k] * nums[k + 1] : nums[k] / nums[k + 1];
      nums[k] = r;
      nums.erase(nums.begin() + k + 1);
      ops.erase(ops.begin() + k);
    } else {
      ++k;
    }
  }
  // Second pass: + and -.
  double acc = nums[0];
  for (std::size_t k = 0; k < ops.size(); ++k)
    acc = (ops[k] == '+') ? acc + nums[k + 1] : acc - nums[k + 1];
  return acc;
}

// Parse "q[i]" -> integer index. Accepts the register name discovered earlier.
int parseQubit(const std::string& tok, const std::string& reg) {
  std::string t = trim(tok);
  std::size_t lb = t.find('[');
  std::size_t rb = t.find(']');
  if (lb == std::string::npos || rb == std::string::npos || rb < lb)
    throw std::runtime_error("qasm: bad qubit operand: " + tok);
  std::string name = trim(t.substr(0, lb));
  if (!reg.empty() && name != reg)
    throw std::runtime_error("qasm: unknown register: " + name);
  return std::stoi(t.substr(lb + 1, rb - lb - 1));
}

}  // namespace

Circuit parseQasm(const std::string& text) {
  std::istringstream in(text);
  std::string raw;
  int nqubits = -1;
  std::string reg;
  std::vector<GateOp> ops;

  // Collect full text, split on ';' so multi-statement lines work too.
  std::string all;
  {
    std::ostringstream ss;
    ss << in.rdbuf();
    all = ss.str();
  }
  // Remove line comments.
  std::string cleaned;
  {
    std::istringstream ls(all);
    std::string line;
    while (std::getline(ls, line)) {
      std::size_t c = line.find("//");
      if (c != std::string::npos) line = line.substr(0, c);
      cleaned += line;
      cleaned += '\n';
    }
  }

  std::stringstream stmts(cleaned);
  std::string stmt;
  while (std::getline(stmts, stmt, ';')) {
    std::string s = trim(stmt);
    if (s.empty()) continue;

    if (s.rfind("OPENQASM", 0) == 0) continue;
    if (s.rfind("include", 0) == 0) continue;

    if (s.rfind("qreg", 0) == 0) {
      std::size_t lb = s.find('[');
      std::size_t rb = s.find(']');
      if (lb == std::string::npos || rb == std::string::npos)
        throw std::runtime_error("qasm: bad qreg: " + s);
      reg = trim(s.substr(4, lb - 4));
      nqubits = std::stoi(s.substr(lb + 1, rb - lb - 1));
      continue;
    }
    if (s.rfind("creg", 0) == 0) continue;
    if (s.rfind("measure", 0) == 0) continue;
    if (s.rfind("barrier", 0) == 0) continue;

    // Gate statement: name[(params)] operand[, operand...]
    std::size_t firstSpace = s.find_first_of(" \t(");
    std::string head =
        (firstSpace == std::string::npos) ? s : s.substr(0, firstSpace);
    std::string rest =
        (firstSpace == std::string::npos) ? "" : s.substr(firstSpace);

    // Extract optional (param) attached to head.
    std::vector<double> params;
    std::size_t lp = s.find('(');
    std::string operandPart;
    std::string name = trim(head);
    if (lp != std::string::npos) {
      std::size_t rp = s.find(')', lp);
      if (rp == std::string::npos)
        throw std::runtime_error("qasm: unmatched '(' in: " + s);
      name = trim(s.substr(0, lp));
      std::string inside = s.substr(lp + 1, rp - lp - 1);
      // Possibly comma-separated params.
      std::stringstream ps(inside);
      std::string p;
      while (std::getline(ps, p, ',')) {
        p = trim(p);
        if (!p.empty()) params.push_back(evalAngle(p));
      }
      operandPart = s.substr(rp + 1);
    } else {
      operandPart = rest;
    }

    // Operands.
    std::vector<int> qubits;
    std::stringstream os(operandPart);
    std::string operand;
    while (std::getline(os, operand, ',')) {
      operand = trim(operand);
      if (operand.empty()) continue;
      qubits.push_back(parseQubit(operand, reg));
    }

    ops.push_back({name, qubits, params});
  }

  if (nqubits < 0)
    throw std::runtime_error("qasm: missing qreg declaration");

  Circuit c(nqubits);
  for (auto& op : ops) c.add(op);
  return c;
}

Circuit parseQasmFile(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("qasm: cannot open file: " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return parseQasm(ss.str());
}

}  // namespace quasar
