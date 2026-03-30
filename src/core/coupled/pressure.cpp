#include "solver.hpp"
#include "internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace ding {

using coupled_detail::square;

namespace {

struct SparseMatrixCSR {
  int n = 0;
  std::vector<int> row_ptr;
  std::vector<int> col_idx;
  std::vector<int> diag_pos;
  std::vector<double> values;
};

struct LowerTransposeEntry {
  int row = 0;
  double value = 0.0;
};

} // namespace

double Solver::solve_pressure_correction_jacobi() {
  pressure_correction_.fill(0.0);
  apply_scalar_bc(pressure_correction_);
  double residual = 0.0;
  last_pressure_iterations_ = 0;
  const double omega = 1.4;

  for (int iter = 0; iter < cfg_.poisson_iterations; ++iter) {
    double max_residual = 0.0;

    apply_scalar_bc(pressure_correction_);
    for (int i = 0; i < cfg_.nx; ++i) {
      for (int j = 0; j < cfg_.ny; ++j) {
        // Non-incremental projection solve for the current pressure field:
        // div( (1/rho) grad(p^{n+1/2}) ) = div(u*) / dt
        // Periodic directions keep the wrapped face coefficients.
        // Non-periodic directions use zero normal derivative for pressure, so the boundary face coefficient is omitted.
        const bool east_open = cfg_.periodic_x || i < cfg_.nx - 1;
        const bool west_open = cfg_.periodic_x || i > 0;
        const bool north_open = cfg_.periodic_y || j < cfg_.ny - 1;
        const bool south_open = cfg_.periodic_y || j > 0;

        const double ae = east_open ? 1.0 / (rho_u_face(rho_mid_, i + 1, j) * dx_ * dx_) : 0.0;
        const double aw = west_open ? 1.0 / (rho_u_face(rho_mid_, i, j) * dx_ * dx_) : 0.0;
        const double an = north_open ? 1.0 / (rho_v_face(rho_mid_, i, j + 1) * dy_ * dy_) : 0.0;
        const double as = south_open ? 1.0 / (rho_v_face(rho_mid_, i, j) * dy_ * dy_) : 0.0;
        const double ap = ae + aw + an + as;
        const double rhs = divergence_cell(u_star_, v_star_, i, j) / cfg_.dt;
        const double residual_local = ae * pressure_correction_(i + 1, j) + aw * pressure_correction_(i - 1, j) +
                                      an * pressure_correction_(i, j + 1) + as * pressure_correction_(i, j - 1) -
                                      ap * pressure_correction_(i, j) - rhs;
        pressure_correction_(i, j) += omega * residual_local / std::max(ap, 1.0e-30);
        max_residual = std::max(max_residual, std::abs(residual_local));
      }
    }

    subtract_mean(pressure_correction_);
    residual = max_residual;
    last_pressure_iterations_ = iter + 1;
    if (cfg_.verbose) {
      std::cout << "[pressure step " << (current_step_index_ + 1) << " outer " << (current_coupling_iteration_ + 1)
                << " iter " << (iter + 1) << "] residual=" << std::scientific << std::setprecision(6) << residual
                << "\n";
    }
    if (residual < cfg_.pressure_tolerance) {
      break;
    }
  }
  apply_scalar_bc(pressure_correction_);
  return residual;
}

double Solver::solve_pressure_correction_icpcg() {
  pressure_correction_.fill(0.0);
  apply_scalar_bc(pressure_correction_);
  last_pressure_iterations_ = 0;

  const int n = cfg_.nx * cfg_.ny;
  SparseMatrixCSR matrix;
  matrix.n = n;
  matrix.row_ptr.reserve(static_cast<std::size_t>(n + 1));
  matrix.diag_pos.resize(static_cast<std::size_t>(n), -1);
  std::vector<std::map<int, double>> row_maps(static_cast<std::size_t>(n));
  std::vector<double> rhs(static_cast<std::size_t>(n), 0.0);

  auto row_index = [&](int i, int j) { return i * cfg_.ny + j; };

  double max_diag = 0.0;
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      const bool east_open = cfg_.periodic_x || i < cfg_.nx - 1;
      const bool west_open = cfg_.periodic_x || i > 0;
      const bool north_open = cfg_.periodic_y || j < cfg_.ny - 1;
      const bool south_open = cfg_.periodic_y || j > 0;
      const double ae = east_open ? 1.0 / (rho_u_face(rho_mid_, i + 1, j) * dx_ * dx_) : 0.0;
      const double aw = west_open ? 1.0 / (rho_u_face(rho_mid_, i, j) * dx_ * dx_) : 0.0;
      const double an = north_open ? 1.0 / (rho_v_face(rho_mid_, i, j + 1) * dy_ * dy_) : 0.0;
      const double as = south_open ? 1.0 / (rho_v_face(rho_mid_, i, j) * dy_ * dy_) : 0.0;
      const double ap = ae + aw + an + as;
      max_diag = std::max(max_diag, ap);

      const int row = row_index(i, j);
      row_maps[static_cast<std::size_t>(row)][row] += ap;
      if (east_open) {
        const int ie = cfg_.periodic_x && i == cfg_.nx - 1 ? 0 : i + 1;
        row_maps[static_cast<std::size_t>(row)][row_index(ie, j)] += -ae;
      }
      if (west_open) {
        const int iw = cfg_.periodic_x && i == 0 ? cfg_.nx - 1 : i - 1;
        row_maps[static_cast<std::size_t>(row)][row_index(iw, j)] += -aw;
      }
      if (north_open) {
        const int jn = cfg_.periodic_y && j == cfg_.ny - 1 ? 0 : j + 1;
        row_maps[static_cast<std::size_t>(row)][row_index(i, jn)] += -an;
      }
      if (south_open) {
        const int js = cfg_.periodic_y && j == 0 ? cfg_.ny - 1 : j - 1;
        row_maps[static_cast<std::size_t>(row)][row_index(i, js)] += -as;
      }
      rhs[static_cast<std::size_t>(row)] = -divergence_cell(u_star_, v_star_, i, j) / cfg_.dt;
    }
  }

  const double rhs_mean =
      std::accumulate(rhs.begin(), rhs.end(), 0.0) / std::max(static_cast<double>(n), 1.0);
  for (double &value : rhs) {
    value -= rhs_mean;
  }

  const double gauge_shift = std::max(1.0, max_diag) * 1.0e-12;
  for (int row = 0; row < n; ++row) {
    row_maps[static_cast<std::size_t>(row)][row] += gauge_shift;
  }

  matrix.row_ptr.push_back(0);
  for (int row = 0; row < n; ++row) {
    for (const auto &[col, value] : row_maps[static_cast<std::size_t>(row)]) {
      matrix.col_idx.push_back(col);
      matrix.values.push_back(value);
      if (col == row) {
        matrix.diag_pos[static_cast<std::size_t>(row)] = static_cast<int>(matrix.values.size()) - 1;
      }
    }
    matrix.row_ptr.push_back(static_cast<int>(matrix.values.size()));
  }

  std::vector<double> lower(matrix.values.size(), 0.0);
  std::vector<double> diag(static_cast<std::size_t>(n), gauge_shift);

  auto find_lower_entry = [&](int row, int target_col) {
    for (int pos = matrix.row_ptr[static_cast<std::size_t>(row)];
         pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos) {
      const int col = matrix.col_idx[static_cast<std::size_t>(pos)];
      if (col >= row || col > target_col) {
        break;
      }
      if (col == target_col) {
        return lower[static_cast<std::size_t>(pos)];
      }
    }
    return 0.0;
  };

  for (int row = 0; row < n; ++row) {
    for (int pos = matrix.row_ptr[static_cast<std::size_t>(row)];
         pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos) {
      const int col = matrix.col_idx[static_cast<std::size_t>(pos)];
      if (col >= row) {
        break;
      }
      double sum = matrix.values[static_cast<std::size_t>(pos)];
      for (int prev = matrix.row_ptr[static_cast<std::size_t>(row)]; prev < pos; ++prev) {
        const int k = matrix.col_idx[static_cast<std::size_t>(prev)];
        if (k >= col) {
          break;
        }
        const double l_jk = find_lower_entry(col, k);
        if (l_jk != 0.0) {
          sum -= lower[static_cast<std::size_t>(prev)] * diag[static_cast<std::size_t>(k)] * l_jk;
        }
      }
      lower[static_cast<std::size_t>(pos)] = sum / std::max(diag[static_cast<std::size_t>(col)], gauge_shift);
    }

    double d = matrix.values[static_cast<std::size_t>(matrix.diag_pos[static_cast<std::size_t>(row)])];
    for (int pos = matrix.row_ptr[static_cast<std::size_t>(row)];
         pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos) {
      const int col = matrix.col_idx[static_cast<std::size_t>(pos)];
      if (col >= row) {
        break;
      }
      d -= square(lower[static_cast<std::size_t>(pos)]) * diag[static_cast<std::size_t>(col)];
    }
    diag[static_cast<std::size_t>(row)] = std::max(d, gauge_shift);
  }

  std::vector<std::vector<LowerTransposeEntry>> lower_transpose(static_cast<std::size_t>(n));
  for (int row = 0; row < n; ++row) {
    for (int pos = matrix.row_ptr[static_cast<std::size_t>(row)];
         pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos) {
      const int col = matrix.col_idx[static_cast<std::size_t>(pos)];
      if (col >= row) {
        break;
      }
      lower_transpose[static_cast<std::size_t>(col)].push_back({row, lower[static_cast<std::size_t>(pos)]});
    }
  }

  auto apply_matrix = [&](const std::vector<double> &x, std::vector<double> &y) {
    std::fill(y.begin(), y.end(), 0.0);
    for (int row = 0; row < n; ++row) {
      double sum = 0.0;
      for (int pos = matrix.row_ptr[static_cast<std::size_t>(row)];
           pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos) {
        sum += matrix.values[static_cast<std::size_t>(pos)] * x[static_cast<std::size_t>(matrix.col_idx[static_cast<std::size_t>(pos)])];
      }
      y[static_cast<std::size_t>(row)] = sum;
    }
  };

  auto apply_preconditioner = [&](const std::vector<double> &r, std::vector<double> &z) {
    std::vector<double> y(static_cast<std::size_t>(n), 0.0);
    for (int row = 0; row < n; ++row) {
      double sum = r[static_cast<std::size_t>(row)];
      for (int pos = matrix.row_ptr[static_cast<std::size_t>(row)];
           pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos) {
        const int col = matrix.col_idx[static_cast<std::size_t>(pos)];
        if (col >= row) {
          break;
        }
        sum -= lower[static_cast<std::size_t>(pos)] * y[static_cast<std::size_t>(col)];
      }
      y[static_cast<std::size_t>(row)] = sum;
    }
    for (int row = 0; row < n; ++row) {
      y[static_cast<std::size_t>(row)] /= std::max(diag[static_cast<std::size_t>(row)], gauge_shift);
    }
    z = y;
    for (int row = n - 1; row >= 0; --row) {
      double sum = z[static_cast<std::size_t>(row)];
      for (const LowerTransposeEntry &entry : lower_transpose[static_cast<std::size_t>(row)]) {
        sum -= entry.value * z[static_cast<std::size_t>(entry.row)];
      }
      z[static_cast<std::size_t>(row)] = sum;
    }
  };

  auto dot_product = [](const std::vector<double> &a, const std::vector<double> &b) {
    double value = 0.0;
    for (std::size_t idx = 0; idx < a.size(); ++idx) {
      value += a[idx] * b[idx];
    }
    return value;
  };

  std::vector<double> x(static_cast<std::size_t>(n), 0.0);
  std::vector<double> r = rhs;
  std::vector<double> z(static_cast<std::size_t>(n), 0.0);
  std::vector<double> p(static_cast<std::size_t>(n), 0.0);
  std::vector<double> ap(static_cast<std::size_t>(n), 0.0);

  const double rhs_norm = std::sqrt(std::max(dot_product(rhs, rhs), 0.0));
  if (rhs_norm < 1.0e-30) {
    if (cfg_.verbose) {
      std::cout << "[pressure step " << (current_step_index_ + 1) << " outer " << (current_coupling_iteration_ + 1)
                << "] [ICPCG] iter 0 residual=0.000000e+00 rel=0.000000e+00\n";
    }
    apply_scalar_bc(pressure_correction_);
    return 0.0;
  }

  apply_preconditioner(r, z);
  p = z;
  double rz_old = dot_product(r, z);
  if (!std::isfinite(rz_old) || rz_old <= 0.0) {
    throw std::runtime_error("ICPCG preconditioner breakdown: non-positive r^T M^{-1} r");
  }

  double residual = std::sqrt(std::max(dot_product(r, r), 0.0));
  double relative_residual = residual / std::max(rhs_norm, 1.0e-30);

  for (int iter = 0; iter < cfg_.poisson_iterations; ++iter) {
    apply_matrix(p, ap);
    const double pap = dot_product(p, ap);
    if (!std::isfinite(pap) || std::abs(pap) < 1.0e-30) {
      throw std::runtime_error("ICPCG breakdown: p^T A p is invalid");
    }

    const double alpha = rz_old / pap;
    for (int idx = 0; idx < n; ++idx) {
      x[static_cast<std::size_t>(idx)] += alpha * p[static_cast<std::size_t>(idx)];
      r[static_cast<std::size_t>(idx)] -= alpha * ap[static_cast<std::size_t>(idx)];
    }

    residual = std::sqrt(std::max(dot_product(r, r), 0.0));
    relative_residual = residual / std::max(rhs_norm, 1.0e-30);
    last_pressure_iterations_ = iter + 1;
    if (cfg_.verbose) {
      std::cout << "[pressure step " << (current_step_index_ + 1) << " outer " << (current_coupling_iteration_ + 1)
                << "] [ICPCG] iter " << (iter + 1) << " residual=" << std::scientific << std::setprecision(6)
                << residual << " rel=" << relative_residual << "\n";
    }
    if (relative_residual < cfg_.pressure_tolerance || residual < cfg_.pressure_tolerance) {
      break;
    }

    apply_preconditioner(r, z);
    const double rz_new = dot_product(r, z);
    if (!std::isfinite(rz_new) || rz_new <= 0.0) {
      throw std::runtime_error("ICPCG preconditioner breakdown: non-positive new r^T M^{-1} r");
    }
    const double beta = rz_new / rz_old;
    for (int idx = 0; idx < n; ++idx) {
      p[static_cast<std::size_t>(idx)] = z[static_cast<std::size_t>(idx)] + beta * p[static_cast<std::size_t>(idx)];
    }
    rz_old = rz_new;
  }

  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      pressure_correction_(i, j) = x[static_cast<std::size_t>(row_index(i, j))];
    }
  }
  subtract_mean(pressure_correction_);
  apply_scalar_bc(pressure_correction_);
  return relative_residual;
}

double Solver::solve_pressure_correction_liu_split_icpcg() {
  pressure_correction_.fill(0.0);
  apply_scalar_bc(pressure_correction_);
  last_pressure_iterations_ = 0;

  Field2D pressure_extrapolated(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0);
  build_liu_split_pressure_extrapolation(pressure_extrapolated);
  const double rho_ref = liu_split_reference_density();

  const int n = cfg_.nx * cfg_.ny;
  SparseMatrixCSR matrix;
  matrix.n = n;
  matrix.row_ptr.reserve(static_cast<std::size_t>(n + 1));
  matrix.diag_pos.resize(static_cast<std::size_t>(n), -1);
  std::vector<std::map<int, double>> row_maps(static_cast<std::size_t>(n));
  std::vector<double> rhs(static_cast<std::size_t>(n), 0.0);

  auto row_index = [&](int i, int j) { return i * cfg_.ny + j; };

  double max_diag = 0.0;
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      const bool east_open = cfg_.periodic_x || i < cfg_.nx - 1;
      const bool west_open = cfg_.periodic_x || i > 0;
      const bool north_open = cfg_.periodic_y || j < cfg_.ny - 1;
      const bool south_open = cfg_.periodic_y || j > 0;
      const double ae = east_open ? 1.0 / (dx_ * dx_) : 0.0;
      const double aw = west_open ? 1.0 / (dx_ * dx_) : 0.0;
      const double an = north_open ? 1.0 / (dy_ * dy_) : 0.0;
      const double as = south_open ? 1.0 / (dy_ * dy_) : 0.0;
      const double ap = ae + aw + an + as;
      max_diag = std::max(max_diag, ap);

      const int row = row_index(i, j);
      row_maps[static_cast<std::size_t>(row)][row] += ap;
      if (east_open) {
        const int ie = cfg_.periodic_x && i == cfg_.nx - 1 ? 0 : i + 1;
        row_maps[static_cast<std::size_t>(row)][row_index(ie, j)] += -ae;
      }
      if (west_open) {
        const int iw = cfg_.periodic_x && i == 0 ? cfg_.nx - 1 : i - 1;
        row_maps[static_cast<std::size_t>(row)][row_index(iw, j)] += -aw;
      }
      if (north_open) {
        const int jn = cfg_.periodic_y && j == cfg_.ny - 1 ? 0 : j + 1;
        row_maps[static_cast<std::size_t>(row)][row_index(i, jn)] += -an;
      }
      if (south_open) {
        const int js = cfg_.periodic_y && j == 0 ? cfg_.ny - 1 : j - 1;
        row_maps[static_cast<std::size_t>(row)][row_index(i, js)] += -as;
      }

      // Liu et al. (2021), Eq. (25):
      // laplacian(p^{n+1/2}) = div[(1-rho_ref/rho_mid) grad(p_ext)] + rho_ref/dt * div(u*)
      const double rhs_continuous =
          liu_split_explicit_divergence(pressure_extrapolated, i, j) + rho_ref * divergence_cell(u_star_, v_star_, i, j) / cfg_.dt;
      rhs[static_cast<std::size_t>(row)] = -rhs_continuous;
    }
  }

  const double rhs_mean =
      std::accumulate(rhs.begin(), rhs.end(), 0.0) / std::max(static_cast<double>(n), 1.0);
  for (double &value : rhs) {
    value -= rhs_mean;
  }

  const double gauge_shift = std::max(1.0, max_diag) * 1.0e-12;
  for (int row = 0; row < n; ++row) {
    row_maps[static_cast<std::size_t>(row)][row] += gauge_shift;
  }

  matrix.row_ptr.push_back(0);
  for (int row = 0; row < n; ++row) {
    for (const auto &[col, value] : row_maps[static_cast<std::size_t>(row)]) {
      matrix.col_idx.push_back(col);
      matrix.values.push_back(value);
      if (col == row) {
        matrix.diag_pos[static_cast<std::size_t>(row)] = static_cast<int>(matrix.values.size()) - 1;
      }
    }
    matrix.row_ptr.push_back(static_cast<int>(matrix.values.size()));
  }

  std::vector<double> lower(matrix.values.size(), 0.0);
  std::vector<double> diag(static_cast<std::size_t>(n), gauge_shift);

  auto find_lower_entry = [&](int row, int target_col) {
    for (int pos = matrix.row_ptr[static_cast<std::size_t>(row)];
         pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos) {
      const int col = matrix.col_idx[static_cast<std::size_t>(pos)];
      if (col >= row || col > target_col) {
        break;
      }
      if (col == target_col) {
        return lower[static_cast<std::size_t>(pos)];
      }
    }
    return 0.0;
  };

  for (int row = 0; row < n; ++row) {
    for (int pos = matrix.row_ptr[static_cast<std::size_t>(row)];
         pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos) {
      const int col = matrix.col_idx[static_cast<std::size_t>(pos)];
      if (col >= row) {
        break;
      }
      double sum = matrix.values[static_cast<std::size_t>(pos)];
      for (int prev = matrix.row_ptr[static_cast<std::size_t>(row)]; prev < pos; ++prev) {
        const int k = matrix.col_idx[static_cast<std::size_t>(prev)];
        if (k >= col) {
          break;
        }
        const double l_jk = find_lower_entry(col, k);
        if (l_jk != 0.0) {
          sum -= lower[static_cast<std::size_t>(prev)] * diag[static_cast<std::size_t>(k)] * l_jk;
        }
      }
      lower[static_cast<std::size_t>(pos)] = sum / std::max(diag[static_cast<std::size_t>(col)], gauge_shift);
    }

    double d = matrix.values[static_cast<std::size_t>(matrix.diag_pos[static_cast<std::size_t>(row)])];
    for (int pos = matrix.row_ptr[static_cast<std::size_t>(row)];
         pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos) {
      const int col = matrix.col_idx[static_cast<std::size_t>(pos)];
      if (col >= row) {
        break;
      }
      d -= square(lower[static_cast<std::size_t>(pos)]) * diag[static_cast<std::size_t>(col)];
    }
    diag[static_cast<std::size_t>(row)] = std::max(d, gauge_shift);
  }

  std::vector<std::vector<LowerTransposeEntry>> lower_transpose(static_cast<std::size_t>(n));
  for (int row = 0; row < n; ++row) {
    for (int pos = matrix.row_ptr[static_cast<std::size_t>(row)];
         pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos) {
      const int col = matrix.col_idx[static_cast<std::size_t>(pos)];
      if (col >= row) {
        break;
      }
      lower_transpose[static_cast<std::size_t>(col)].push_back({row, lower[static_cast<std::size_t>(pos)]});
    }
  }

  auto apply_matrix = [&](const std::vector<double> &x, std::vector<double> &y) {
    std::fill(y.begin(), y.end(), 0.0);
    for (int row = 0; row < n; ++row) {
      double sum = 0.0;
      for (int pos = matrix.row_ptr[static_cast<std::size_t>(row)];
           pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos) {
        sum += matrix.values[static_cast<std::size_t>(pos)] *
               x[static_cast<std::size_t>(matrix.col_idx[static_cast<std::size_t>(pos)])];
      }
      y[static_cast<std::size_t>(row)] = sum;
    }
  };

  auto apply_preconditioner = [&](const std::vector<double> &r, std::vector<double> &z) {
    std::vector<double> y(static_cast<std::size_t>(n), 0.0);
    for (int row = 0; row < n; ++row) {
      double sum = r[static_cast<std::size_t>(row)];
      for (int pos = matrix.row_ptr[static_cast<std::size_t>(row)];
           pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos) {
        const int col = matrix.col_idx[static_cast<std::size_t>(pos)];
        if (col >= row) {
          break;
        }
        sum -= lower[static_cast<std::size_t>(pos)] * y[static_cast<std::size_t>(col)];
      }
      y[static_cast<std::size_t>(row)] = sum;
    }
    for (int row = 0; row < n; ++row) {
      y[static_cast<std::size_t>(row)] /= std::max(diag[static_cast<std::size_t>(row)], gauge_shift);
    }
    z = y;
    for (int row = n - 1; row >= 0; --row) {
      double sum = z[static_cast<std::size_t>(row)];
      for (const LowerTransposeEntry &entry : lower_transpose[static_cast<std::size_t>(row)]) {
        sum -= entry.value * z[static_cast<std::size_t>(entry.row)];
      }
      z[static_cast<std::size_t>(row)] = sum;
    }
  };

  auto dot_product = [](const std::vector<double> &a, const std::vector<double> &b) {
    double value = 0.0;
    for (std::size_t idx = 0; idx < a.size(); ++idx) {
      value += a[idx] * b[idx];
    }
    return value;
  };

  std::vector<double> x(static_cast<std::size_t>(n), 0.0);
  std::vector<double> r = rhs;
  std::vector<double> z(static_cast<std::size_t>(n), 0.0);
  std::vector<double> p(static_cast<std::size_t>(n), 0.0);
  std::vector<double> ap(static_cast<std::size_t>(n), 0.0);

  const double rhs_norm = std::sqrt(std::max(dot_product(rhs, rhs), 0.0));
  if (rhs_norm < 1.0e-30) {
    if (cfg_.verbose) {
      std::cout << "[pressure step " << (current_step_index_ + 1) << " outer " << (current_coupling_iteration_ + 1)
                << "] [LiuSplitICPCG] iter 0 residual=0.000000e+00 rel=0.000000e+00\n";
    }
    apply_scalar_bc(pressure_correction_);
    return 0.0;
  }

  apply_preconditioner(r, z);
  p = z;
  double rz_old = dot_product(r, z);
  if (!std::isfinite(rz_old) || rz_old <= 0.0) {
    throw std::runtime_error("Liu split ICPCG preconditioner breakdown: non-positive r^T M^{-1} r");
  }

  double residual = std::sqrt(std::max(dot_product(r, r), 0.0));
  double relative_residual = residual / std::max(rhs_norm, 1.0e-30);

  for (int iter = 0; iter < cfg_.poisson_iterations; ++iter) {
    apply_matrix(p, ap);
    const double pap = dot_product(p, ap);
    if (!std::isfinite(pap) || std::abs(pap) < 1.0e-30) {
      throw std::runtime_error("Liu split ICPCG breakdown: p^T A p is invalid");
    }

    const double alpha = rz_old / pap;
    for (int idx = 0; idx < n; ++idx) {
      x[static_cast<std::size_t>(idx)] += alpha * p[static_cast<std::size_t>(idx)];
      r[static_cast<std::size_t>(idx)] -= alpha * ap[static_cast<std::size_t>(idx)];
    }

    residual = std::sqrt(std::max(dot_product(r, r), 0.0));
    relative_residual = residual / std::max(rhs_norm, 1.0e-30);
    last_pressure_iterations_ = iter + 1;
    if (cfg_.verbose) {
      std::cout << "[pressure step " << (current_step_index_ + 1) << " outer " << (current_coupling_iteration_ + 1)
                << "] [LiuSplitICPCG] iter " << (iter + 1) << " residual=" << std::scientific
                << std::setprecision(6) << residual << " rel=" << relative_residual << "\n";
    }
    if (relative_residual < cfg_.pressure_tolerance || residual < cfg_.pressure_tolerance) {
      break;
    }

    apply_preconditioner(r, z);
    const double rz_new = dot_product(r, z);
    if (!std::isfinite(rz_new) || rz_new <= 0.0) {
      throw std::runtime_error("Liu split ICPCG preconditioner breakdown: non-positive new r^T M^{-1} r");
    }
    const double beta = rz_new / rz_old;
    for (int idx = 0; idx < n; ++idx) {
      p[static_cast<std::size_t>(idx)] = z[static_cast<std::size_t>(idx)] + beta * p[static_cast<std::size_t>(idx)];
    }
    rz_old = rz_new;
  }

  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      pressure_correction_(i, j) = x[static_cast<std::size_t>(row_index(i, j))];
    }
  }
  subtract_mean(pressure_correction_);
  apply_scalar_bc(pressure_correction_);
  return relative_residual;
}

double Solver::solve_pressure_correction_petsc() {
  namespace fs = std::filesystem;
  last_pressure_iterations_ = 0;

  const fs::path solver_dir = pressure_solver_dir();
  fs::create_directories(solver_dir);

  const fs::path matrix_path = solver_dir / "matrix_triplets.txt";
  const fs::path rhs_path = solver_dir / "rhs.txt";
  const fs::path solution_path = solver_dir / "solution.txt";
  const fs::path report_path = solver_dir / "report.txt";
  const bool write_monitor_log =
      cfg_.petsc_pressure_log_every > 0 && ((current_step_index_ + 1) % cfg_.petsc_pressure_log_every == 0);
  std::ostringstream monitor_name;
  monitor_name << "residual_step_" << std::setw(6) << std::setfill('0') << (current_step_index_ + 1) << ".log";
  const fs::path monitor_log_path = solver_dir / monitor_name.str();

  const int n = cfg_.nx * cfg_.ny;
  std::vector<double> rhs_values(static_cast<std::size_t>(n), 0.0);
  std::size_t nnz = 0;

  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      const bool east_open = cfg_.periodic_x || i < cfg_.nx - 1;
      const bool west_open = cfg_.periodic_x || i > 0;
      const bool north_open = cfg_.periodic_y || j < cfg_.ny - 1;
      const bool south_open = cfg_.periodic_y || j > 0;
      const double ae = east_open ? 1.0 / (rho_u_face(rho_mid_, i + 1, j) * dx_ * dx_) : 0.0;
      const double aw = west_open ? 1.0 / (rho_u_face(rho_mid_, i, j) * dx_ * dx_) : 0.0;
      const double an = north_open ? 1.0 / (rho_v_face(rho_mid_, i, j + 1) * dy_ * dy_) : 0.0;
      const double as = south_open ? 1.0 / (rho_v_face(rho_mid_, i, j) * dy_ * dy_) : 0.0;
      const double ap = ae + aw + an + as;
        const int rhs_index = i * cfg_.ny + j;
        rhs_values[static_cast<std::size_t>(rhs_index)] = -divergence_cell(u_star_, v_star_, i, j) / cfg_.dt;
      nnz += 1;
      if (east_open) {
        ++nnz;
      }
      if (west_open) {
        ++nnz;
      }
      if (north_open) {
        ++nnz;
      }
      if (south_open) {
        ++nnz;
      }
      (void)ap;
    }
  }

  {
    std::ofstream matrix_out(matrix_path);
    if (!matrix_out) {
      throw std::runtime_error("cannot open PETSc pressure matrix file: " + matrix_path.string());
    }
    matrix_out << n << " " << n << " " << nnz << "\n";
    matrix_out << std::setprecision(17);
    for (int i = 0; i < cfg_.nx; ++i) {
      for (int j = 0; j < cfg_.ny; ++j) {
        const bool east_open = cfg_.periodic_x || i < cfg_.nx - 1;
        const bool west_open = cfg_.periodic_x || i > 0;
        const bool north_open = cfg_.periodic_y || j < cfg_.ny - 1;
        const bool south_open = cfg_.periodic_y || j > 0;
        const double ae = east_open ? 1.0 / (rho_u_face(rho_mid_, i + 1, j) * dx_ * dx_) : 0.0;
        const double aw = west_open ? 1.0 / (rho_u_face(rho_mid_, i, j) * dx_ * dx_) : 0.0;
        const double an = north_open ? 1.0 / (rho_v_face(rho_mid_, i, j + 1) * dy_ * dy_) : 0.0;
        const double as = south_open ? 1.0 / (rho_v_face(rho_mid_, i, j) * dy_ * dy_) : 0.0;
        const double ap = ae + aw + an + as;
        const int row = i * cfg_.ny + j;

        matrix_out << row << " " << row << " " << ap << "\n";
        if (east_open) {
          const int ie = cfg_.periodic_x && i == cfg_.nx - 1 ? 0 : i + 1;
          matrix_out << row << " " << (ie * cfg_.ny + j) << " " << -ae << "\n";
        }
        if (west_open) {
          const int iw = cfg_.periodic_x && i == 0 ? cfg_.nx - 1 : i - 1;
          matrix_out << row << " " << (iw * cfg_.ny + j) << " " << -aw << "\n";
        }
        if (north_open) {
          const int jn = cfg_.periodic_y && j == cfg_.ny - 1 ? 0 : j + 1;
          matrix_out << row << " " << (i * cfg_.ny + jn) << " " << -an << "\n";
        }
        if (south_open) {
          const int js = cfg_.periodic_y && j == 0 ? cfg_.ny - 1 : j - 1;
          matrix_out << row << " " << (i * cfg_.ny + js) << " " << -as << "\n";
        }
      }
    }
  }

  {
    std::ofstream rhs_out(rhs_path);
    if (!rhs_out) {
      throw std::runtime_error("cannot open PETSc pressure rhs file: " + rhs_path.string());
    }
    rhs_out << n << "\n";
    rhs_out << std::setprecision(17);
    for (double value : rhs_values) {
      rhs_out << value << "\n";
    }
  }

  const fs::path script_path = cfg_.petsc_solver_script;
  const fs::path options_path = cfg_.petsc_solver_config;
  if (!fs::exists(script_path)) {
    throw std::runtime_error("PETSc pressure helper script not found: " + script_path.string());
  }
  if (!fs::exists(options_path)) {
    throw std::runtime_error("PETSc pressure options file not found: " + options_path.string());
  }
  std::ostringstream command;
  if (const char *petsc_dir = std::getenv("PETSC_DIR")) {
    if (*petsc_dir != '\0') {
      command << "PETSC_DIR=" << petsc_dir << " ";
    }
  }
  if (const char *petsc_arch = std::getenv("PETSC_ARCH")) {
    if (*petsc_arch != '\0') {
      command << "PETSC_ARCH=" << petsc_arch << " ";
    }
  }
  command << cfg_.petsc_python_executable << " " << script_path.string() << " --matrix " << matrix_path.string()
          << " --rhs " << rhs_path.string() << " --solution " << solution_path.string() << " --report "
          << report_path.string() << " --config " << options_path.string();
  if (cfg_.verbose) {
    command << " --log-prefix \"[pressure step " << (current_step_index_ + 1) << " outer "
            << (current_coupling_iteration_ + 1) << "]\"";
  }
  if (write_monitor_log) {
    command << " --monitor-log " << monitor_log_path.string();
  }

  const int code = std::system(command.str().c_str());
  if (code != 0) {
    throw std::runtime_error("PETSc pressure solve failed with exit code " + std::to_string(code));
  }

  {
    std::ifstream solution_in(solution_path);
    if (!solution_in) {
      throw std::runtime_error("cannot open PETSc pressure solution file: " + solution_path.string());
    }
    int read_n = 0;
    solution_in >> read_n;
    if (read_n != n) {
      throw std::runtime_error("PETSc pressure solution size mismatch");
    }
    for (int i = 0; i < cfg_.nx; ++i) {
      for (int j = 0; j < cfg_.ny; ++j) {
        solution_in >> pressure_correction_(i, j);
      }
    }
  }

  double residual = 0.0;
  {
    std::ifstream report_in(report_path);
    if (!report_in) {
      throw std::runtime_error("cannot open PETSc pressure report file: " + report_path.string());
    }
    std::string key;
    while (report_in >> key) {
      if (key == "residual_norm") {
        report_in >> residual;
      } else if (key == "iterations") {
        report_in >> last_pressure_iterations_;
      } else {
        std::string value;
        report_in >> value;
      }
    }
  }

  subtract_mean(pressure_correction_);
  apply_scalar_bc(pressure_correction_);
  return residual;
}

double Solver::solve_pressure_correction_hydea() {
  namespace fs = std::filesystem;
  last_pressure_iterations_ = 0;

  const fs::path solver_dir = pressure_solver_dir();
  fs::create_directories(solver_dir);

  const fs::path matrix_path = solver_dir / "matrix_triplets.txt";
  const fs::path rhs_path = solver_dir / "rhs.txt";
  const fs::path solution_path = solver_dir / "solution.txt";
  const fs::path report_path = solver_dir / "report.txt";
  const bool write_monitor_log =
      cfg_.petsc_pressure_log_every > 0 && ((current_step_index_ + 1) % cfg_.petsc_pressure_log_every == 0);
  std::ostringstream monitor_name;
  monitor_name << "residual_step_" << std::setw(6) << std::setfill('0') << (current_step_index_ + 1) << ".log";
  const fs::path monitor_log_path = solver_dir / monitor_name.str();

  const int n = cfg_.nx * cfg_.ny;
  std::vector<double> rhs_values(static_cast<std::size_t>(n), 0.0);
  std::size_t nnz = 0;

  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      const bool east_open = cfg_.periodic_x || i < cfg_.nx - 1;
      const bool west_open = cfg_.periodic_x || i > 0;
      const bool north_open = cfg_.periodic_y || j < cfg_.ny - 1;
      const bool south_open = cfg_.periodic_y || j > 0;
      const int rhs_index = i * cfg_.ny + j;
      rhs_values[static_cast<std::size_t>(rhs_index)] = -divergence_cell(u_star_, v_star_, i, j) / cfg_.dt;
      nnz += 1;
      if (east_open) ++nnz;
      if (west_open) ++nnz;
      if (north_open) ++nnz;
      if (south_open) ++nnz;
    }
  }

  {
    std::ofstream matrix_out(matrix_path);
    if (!matrix_out) {
      throw std::runtime_error("cannot open HyDEA pressure matrix file: " + matrix_path.string());
    }
    matrix_out << n << " " << n << " " << nnz << "\n";
    matrix_out << std::setprecision(17);
    for (int i = 0; i < cfg_.nx; ++i) {
      for (int j = 0; j < cfg_.ny; ++j) {
        const bool east_open = cfg_.periodic_x || i < cfg_.nx - 1;
        const bool west_open = cfg_.periodic_x || i > 0;
        const bool north_open = cfg_.periodic_y || j < cfg_.ny - 1;
        const bool south_open = cfg_.periodic_y || j > 0;
        const double ae = east_open ? 1.0 / (rho_u_face(rho_mid_, i + 1, j) * dx_ * dx_) : 0.0;
        const double aw = west_open ? 1.0 / (rho_u_face(rho_mid_, i, j) * dx_ * dx_) : 0.0;
        const double an = north_open ? 1.0 / (rho_v_face(rho_mid_, i, j + 1) * dy_ * dy_) : 0.0;
        const double as = south_open ? 1.0 / (rho_v_face(rho_mid_, i, j) * dy_ * dy_) : 0.0;
        const double ap = ae + aw + an + as;
        const int row = i * cfg_.ny + j;

        matrix_out << row << " " << row << " " << ap << "\n";
        if (east_open) {
          const int ie = cfg_.periodic_x && i == cfg_.nx - 1 ? 0 : i + 1;
          matrix_out << row << " " << (ie * cfg_.ny + j) << " " << -ae << "\n";
        }
        if (west_open) {
          const int iw = cfg_.periodic_x && i == 0 ? cfg_.nx - 1 : i - 1;
          matrix_out << row << " " << (iw * cfg_.ny + j) << " " << -aw << "\n";
        }
        if (north_open) {
          const int jn = cfg_.periodic_y && j == cfg_.ny - 1 ? 0 : j + 1;
          matrix_out << row << " " << (i * cfg_.ny + jn) << " " << -an << "\n";
        }
        if (south_open) {
          const int js = cfg_.periodic_y && j == 0 ? cfg_.ny - 1 : j - 1;
          matrix_out << row << " " << (i * cfg_.ny + js) << " " << -as << "\n";
        }
      }
    }
  }

  {
    std::ofstream rhs_out(rhs_path);
    if (!rhs_out) {
      throw std::runtime_error("cannot open HyDEA pressure rhs file: " + rhs_path.string());
    }
    rhs_out << n << "\n";
    rhs_out << std::setprecision(17);
    for (double value : rhs_values) {
      rhs_out << value << "\n";
    }
  }

  const fs::path script_path = cfg_.hydea_solver_script;
  const fs::path options_path = cfg_.hydea_solver_config;
  if (!fs::exists(script_path)) {
    throw std::runtime_error("HyDEA pressure helper script not found: " + script_path.string());
  }
  if (!fs::exists(options_path)) {
    throw std::runtime_error("HyDEA pressure options file not found: " + options_path.string());
  }
  std::ostringstream command;
  if (const char *petsc_dir = std::getenv("PETSC_DIR")) {
    if (*petsc_dir != '\0') {
      command << "PETSC_DIR=" << petsc_dir << " ";
    }
  }
  if (const char *petsc_arch = std::getenv("PETSC_ARCH")) {
    if (*petsc_arch != '\0') {
      command << "PETSC_ARCH=" << petsc_arch << " ";
    }
  }
  command << cfg_.petsc_python_executable << " " << script_path.string()
          << " --matrix " << matrix_path.string()
          << " --rhs " << rhs_path.string()
          << " --solution " << solution_path.string()
          << " --report " << report_path.string()
          << " --config " << options_path.string()
          << " --model-path " << cfg_.hydea_model_path
          << " --grid-nx " << cfg_.nx
          << " --grid-ny " << cfg_.ny;
  if (cfg_.verbose) {
    command << " --log-prefix \"[pressure step " << (current_step_index_ + 1) << " outer "
            << (current_coupling_iteration_ + 1) << "]\"";
  }
  if (write_monitor_log) {
    command << " --monitor-log " << monitor_log_path.string();
  }

  const int code = std::system(command.str().c_str());
  if (code != 0) {
    throw std::runtime_error("HyDEA pressure solve failed with exit code " + std::to_string(code));
  }

  {
    std::ifstream solution_in(solution_path);
    if (!solution_in) {
      throw std::runtime_error("cannot open HyDEA pressure solution file: " + solution_path.string());
    }
    int read_n = 0;
    solution_in >> read_n;
    if (read_n != n) {
      throw std::runtime_error("HyDEA pressure solution size mismatch");
    }
    for (int i = 0; i < cfg_.nx; ++i) {
      for (int j = 0; j < cfg_.ny; ++j) {
        solution_in >> pressure_correction_(i, j);
      }
    }
  }

  double residual = 0.0;
  {
    std::ifstream report_in(report_path);
    if (!report_in) {
      throw std::runtime_error("cannot open HyDEA pressure report file: " + report_path.string());
    }
    std::string key;
    while (report_in >> key) {
      if (key == "residual_norm") {
        report_in >> residual;
      } else if (key == "iterations") {
        report_in >> last_pressure_iterations_;
      } else {
        std::string value;
        report_in >> value;
      }
    }
  }

  subtract_mean(pressure_correction_);
  apply_scalar_bc(pressure_correction_);
  return residual;
}

double Solver::solve_pressure_correction() {
  const char *solver_name = "Jacobi";
  double residual = 0.0;
  if (cfg_.pressure_scheme == "petsc_pcg") {
    solver_name = "PETSc";
    residual = solve_pressure_correction_petsc();
  } else if (cfg_.pressure_scheme == "hydea") {
    solver_name = "HyDEA";
    residual = solve_pressure_correction_hydea();
  } else if (use_liu_pressure_split()) {
    solver_name = "LiuSplitICPCG";
    residual = solve_pressure_correction_liu_split_icpcg();
  } else if (cfg_.pressure_scheme == "icpcg") {
    solver_name = "ICPCG";
    residual = solve_pressure_correction_icpcg();
  } else {
    residual = solve_pressure_correction_jacobi();
  }
  if (cfg_.verbose) {
    std::cout << "[pressure step " << (current_step_index_ + 1) << " outer " << (current_coupling_iteration_ + 1)
              << "] [" << solver_name << "] done iterations=" << last_pressure_iterations_ << " residual="
              << std::scientific << std::setprecision(6) << residual << "\n";
  }
  return residual;
}


} // namespace ding
