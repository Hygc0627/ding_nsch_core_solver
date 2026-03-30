#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace ding {

struct Field2D {
  int nx = 0;
  int ny = 0;
  int ghost = 0;
  std::vector<double> data;

  Field2D() = default;

  Field2D(int nx_, int ny_, int ghost_, double value = 0.0)
      : nx(nx_),
        ny(ny_),
        ghost(ghost_),
        data(static_cast<std::size_t>(nx_ + 2 * ghost_) * static_cast<std::size_t>(ny_ + 2 * ghost_), value) {}

  double &operator()(int i, int j) {
    const int ii = i + ghost;
    const int jj = j + ghost;
    return data[static_cast<std::size_t>(ii) * static_cast<std::size_t>(ny + 2 * ghost) + static_cast<std::size_t>(jj)];
  }

  double operator()(int i, int j) const {
    const int ii = i + ghost;
    const int jj = j + ghost;
    return data[static_cast<std::size_t>(ii) * static_cast<std::size_t>(ny + 2 * ghost) + static_cast<std::size_t>(jj)];
  }

  void fill(double value) { std::fill(data.begin(), data.end(), value); }
};

} // namespace ding
