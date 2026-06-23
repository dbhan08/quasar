#include "quasar/golden.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace quasar {

namespace {

// Minimal JSON scanning for the golden file: pull a scalar value by key, and
// pull a flat numeric array by key. The golden file is machine-generated with a
// flat structure, so this targeted approach is sufficient.

std::string readFile(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("golden: cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::size_t findKey(const std::string& s, const std::string& key) {
  std::string needle = "\"" + key + "\"";
  std::size_t p = s.find(needle);
  if (p == std::string::npos)
    throw std::runtime_error("golden: missing key " + key);
  p = s.find(':', p + needle.size());
  if (p == std::string::npos)
    throw std::runtime_error("golden: malformed key " + key);
  return p + 1;
}

double readNumberAt(const std::string& s, std::size_t pos) {
  while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos])))
    ++pos;
  std::size_t start = pos;
  while (pos < s.size() &&
         (std::isdigit(static_cast<unsigned char>(s[pos])) || s[pos] == '-' ||
          s[pos] == '+' || s[pos] == '.' || s[pos] == 'e' || s[pos] == 'E'))
    ++pos;
  return std::stod(s.substr(start, pos - start));
}

std::string readStringAt(const std::string& s, std::size_t pos) {
  while (pos < s.size() && s[pos] != '"') ++pos;
  ++pos;
  std::size_t start = pos;
  while (pos < s.size() && s[pos] != '"') ++pos;
  return s.substr(start, pos - start);
}

std::vector<double> readArrayAt(const std::string& s, std::size_t pos) {
  while (pos < s.size() && s[pos] != '[') ++pos;
  ++pos;
  std::vector<double> out;
  while (pos < s.size() && s[pos] != ']') {
    while (pos < s.size() &&
           (std::isspace(static_cast<unsigned char>(s[pos])) || s[pos] == ','))
      ++pos;
    if (pos >= s.size() || s[pos] == ']') break;
    std::size_t start = pos;
    while (pos < s.size() && s[pos] != ',' && s[pos] != ']') ++pos;
    out.push_back(std::stod(s.substr(start, pos - start)));
  }
  return out;
}

}  // namespace

Golden loadGoldenFile(const std::string& path) {
  std::string s = readFile(path);
  Golden g;
  g.nqubits = static_cast<int>(readNumberAt(s, findKey(s, "nqubits")));
  g.circuit = readStringAt(s, findKey(s, "circuit"));
  g.baseline = readStringAt(s, findKey(s, "baseline"));
  g.aer_time_s = readNumberAt(s, findKey(s, "aer_time_s"));
  std::vector<double> re = readArrayAt(s, findKey(s, "re"));
  std::vector<double> im = readArrayAt(s, findKey(s, "im"));
  if (re.size() != im.size())
    throw std::runtime_error("golden: re/im length mismatch");
  g.state.resize(re.size());
  for (std::size_t i = 0; i < re.size(); ++i) g.state[i] = cd(re[i], im[i]);
  return g;
}

}  // namespace quasar
