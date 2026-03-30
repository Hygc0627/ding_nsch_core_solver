#pragma once

#include "core/base/config.hpp"

#include <algorithm>

namespace ding {

struct UniformGrid2D {
  int nx = 0;
  int ny = 0;
  int ghost = 0;
  double lx = 1.0;
  double ly = 1.0;
  double dx = 0.0;
  double dy = 0.0;

  UniformGrid2D() = default;

  explicit UniformGrid2D(const Config &cfg)
      : nx(cfg.nx),
        ny(cfg.ny),
        ghost(cfg.ghost),
        lx(cfg.lx),
        ly(cfg.ly),
        dx(cfg.lx / std::max(static_cast<double>(cfg.nx), 1.0)),
        dy(cfg.ly / std::max(static_cast<double>(cfg.ny), 1.0)) {}
};

} // namespace ding
