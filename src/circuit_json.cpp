#include "quasar/circuit_json.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace quasar {

namespace {

// A tiny recursive-descent JSON reader supporting objects, arrays, strings,
// numbers, and the literals needed for our circuit format. Not a general JSON
// parser; it is just enough for the harness output.
struct Reader {
  const std::string& s;
  std::size_t i = 0;
  explicit Reader(const std::string& str) : s(str) {}

  void ws() {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
  }
  char peek() {
    ws();
    if (i >= s.size()) throw std::runtime_error("json: unexpected end");
    return s[i];
  }
  void expect(char c) {
    ws();
    if (i >= s.size() || s[i] != c)
      throw std::runtime_error(std::string("json: expected '") + c + "'");
    ++i;
  }
  std::string readString() {
    expect('"');
    std::string out;
    while (i < s.size() && s[i] != '"') {
      if (s[i] == '\\' && i + 1 < s.size()) {
        ++i;
        out += s[i];
      } else {
        out += s[i];
      }
      ++i;
    }
    expect('"');
    return out;
  }
  double readNumber() {
    ws();
    std::size_t start = i;
    while (i < s.size() &&
           (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '-' ||
            s[i] == '+' || s[i] == '.' || s[i] == 'e' || s[i] == 'E'))
      ++i;
    if (start == i) throw std::runtime_error("json: bad number");
    return std::stod(s.substr(start, i - start));
  }
};

}  // namespace

Circuit loadCircuitJson(const std::string& text) {
  Reader r(text);
  r.expect('{');
  int nqubits = -1;
  std::vector<GateOp> ops;

  bool first = true;
  while (true) {
    if (r.peek() == '}') {
      ++r.i;
      break;
    }
    if (!first) r.expect(',');
    first = false;
    std::string key = r.readString();
    r.expect(':');
    if (key == "nqubits") {
      nqubits = static_cast<int>(r.readNumber());
    } else if (key == "ops") {
      r.expect('[');
      if (r.peek() == ']') {
        ++r.i;
      } else {
        bool firstOp = true;
        while (true) {
          if (!firstOp) r.expect(',');
          firstOp = false;
          // Parse one op object.
          r.expect('{');
          GateOp op;
          bool firstField = true;
          while (true) {
            if (r.peek() == '}') {
              ++r.i;
              break;
            }
            if (!firstField) r.expect(',');
            firstField = false;
            std::string fk = r.readString();
            r.expect(':');
            if (fk == "name") {
              op.name = r.readString();
            } else if (fk == "qubits") {
              r.expect('[');
              if (r.peek() == ']') {
                ++r.i;
              } else {
                bool fe = true;
                while (true) {
                  if (!fe) r.expect(',');
                  fe = false;
                  op.qubits.push_back(static_cast<int>(r.readNumber()));
                  if (r.peek() == ']') {
                    ++r.i;
                    break;
                  }
                }
              }
            } else if (fk == "params") {
              r.expect('[');
              if (r.peek() == ']') {
                ++r.i;
              } else {
                bool fe = true;
                while (true) {
                  if (!fe) r.expect(',');
                  fe = false;
                  op.params.push_back(r.readNumber());
                  if (r.peek() == ']') {
                    ++r.i;
                    break;
                  }
                }
              }
            } else {
              throw std::runtime_error("json: unknown op field: " + fk);
            }
          }
          ops.push_back(std::move(op));
          if (r.peek() == ']') {
            ++r.i;
            break;
          }
        }
      }
    } else {
      throw std::runtime_error("json: unknown top-level key: " + key);
    }
  }

  if (nqubits < 0) throw std::runtime_error("json: missing nqubits");
  Circuit c(nqubits);
  for (auto& op : ops) c.add(op);
  return c;
}

Circuit loadCircuitJsonFile(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("json: cannot open file: " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return loadCircuitJson(ss.str());
}

}  // namespace quasar
