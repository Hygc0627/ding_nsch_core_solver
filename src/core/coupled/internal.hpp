#pragma once

#include <algorithm>
#include <cctype>
#include <string>

namespace ding::coupled_detail {

inline std::string trim(const std::string &text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1);
}

inline bool parse_bool(const std::string &value) {
  std::string lower = value;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return lower == "1" || lower == "true" || lower == "yes" || lower == "on";
}

inline double square(double value) { return value * value; }

} // namespace ding::coupled_detail
