#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <stdexcept>
#include <vector>

namespace ding::ch_sparse_krylov {

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

enum class PreconditionerType {
  diagonal,
  incomplete_ldlt,
  incomplete_cholesky,
};

struct KrylovPreconditioner {
  PreconditionerType type = PreconditionerType::diagonal;
  std::vector<double> diagonal_inverse;
  std::vector<double> lower;
  std::vector<double> diag;
  std::vector<std::vector<LowerTransposeEntry>> lower_transpose;
  double diagonal_floor = 1.0e-30;
};

struct LinearSolveReport {
  int iterations = 0;
  double absolute_residual = 0.0;
  double relative_residual = 0.0;
  double iterate_residual = 0.0;
};

struct CGMonitorData {
  int iteration = 0;
  double alpha = 0.0;
  double beta = 0.0;
  double absolute_residual = 0.0;
  double relative_residual = 0.0;
  double iterate_residual = 0.0;
  double preconditioned_residual = 0.0;
  const std::vector<double> *x = nullptr;
  const std::vector<double> *r = nullptr;
  const std::vector<double> *z = nullptr;
};

using VectorProjection = std::function<void(std::vector<double> &)>;
using CGMonitor = std::function<void(const CGMonitorData &)>;

struct DirectionGenerationContext {
  int iteration = 0;
  double residual_norm = 0.0;
  const std::vector<double> *x = nullptr;
  const std::vector<double> *r = nullptr;
  const std::vector<double> *rhs = nullptr;
};

using DirectionGenerator =
    std::function<bool(const std::vector<double> &normalized_residual, std::vector<double> &candidate_direction,
                       const DirectionGenerationContext &context)>;

struct DCDMOptions {
  int history_size = 2;
  int max_stored_directions = 8;
  int restart_interval = 0;
  double direction_floor = 1.0e-30;
};

inline double square(double value) { return value * value; }

inline double dot_product(const std::vector<double> &a, const std::vector<double> &b) {
  double value = 0.0;
  for (std::size_t idx = 0; idx < a.size(); ++idx) {
    value += a[idx] * b[idx];
  }
  return value;
}

inline double l2_norm(const std::vector<double> &values) {
  return std::sqrt(std::max(dot_product(values, values), 0.0));
}

inline bool vector_is_finite(const std::vector<double> &values) {
  for (double value : values) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

inline void finalize_row_maps(const std::vector<std::map<int, double>> &row_maps, SparseMatrixCSR &matrix) {
  matrix.n = static_cast<int>(row_maps.size());
  matrix.row_ptr.clear();
  matrix.col_idx.clear();
  matrix.values.clear();
  matrix.diag_pos.assign(static_cast<std::size_t>(matrix.n), -1);
  matrix.row_ptr.reserve(static_cast<std::size_t>(matrix.n + 1));
  matrix.row_ptr.push_back(0);

  for (int row = 0; row < matrix.n; ++row) {
    for (const auto &[col, value] : row_maps[static_cast<std::size_t>(row)]) {
      if (std::abs(value) < 1.0e-18) {
        continue;
      }
      matrix.col_idx.push_back(col);
      matrix.values.push_back(value);
      if (col == row) {
        matrix.diag_pos[static_cast<std::size_t>(row)] = static_cast<int>(matrix.values.size()) - 1;
      }
    }
    matrix.row_ptr.push_back(static_cast<int>(matrix.values.size()));
  }

  for (int row = 0; row < matrix.n; ++row) {
    if (matrix.diag_pos[static_cast<std::size_t>(row)] < 0) {
      throw std::runtime_error("sparse matrix assembly failed to produce a diagonal entry");
    }
  }
}

inline SparseMatrixCSR build_laplacian_matrix(int nx, int ny, double dx, double dy, bool periodic_x, bool periodic_y) {
  const int n = nx * ny;
  const double inv_dx2 = 1.0 / (dx * dx);
  const double inv_dy2 = 1.0 / (dy * dy);
  std::vector<std::map<int, double>> row_maps(static_cast<std::size_t>(n));

  auto row_index = [ny](int i, int j) { return i * ny + j; };

  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      const int row = row_index(i, j);
      double diag = 0.0;

      if (periodic_x || i > 0) {
        const int iw = periodic_x && i == 0 ? nx - 1 : i - 1;
        row_maps[static_cast<std::size_t>(row)][row_index(iw, j)] += inv_dx2;
        diag -= inv_dx2;
      }
      if (periodic_x || i < nx - 1) {
        const int ie = periodic_x && i == nx - 1 ? 0 : i + 1;
        row_maps[static_cast<std::size_t>(row)][row_index(ie, j)] += inv_dx2;
        diag -= inv_dx2;
      }

      if (periodic_y || j > 0) {
        const int js = periodic_y && j == 0 ? ny - 1 : j - 1;
        row_maps[static_cast<std::size_t>(row)][row_index(i, js)] += inv_dy2;
        diag -= inv_dy2;
      }
      if (periodic_y || j < ny - 1) {
        const int jn = periodic_y && j == ny - 1 ? 0 : j + 1;
        row_maps[static_cast<std::size_t>(row)][row_index(i, jn)] += inv_dy2;
        diag -= inv_dy2;
      }

      row_maps[static_cast<std::size_t>(row)][row] += diag;
    }
  }

  SparseMatrixCSR matrix;
  finalize_row_maps(row_maps, matrix);
  return matrix;
}

inline SparseMatrixCSR multiply(const SparseMatrixCSR &a, const SparseMatrixCSR &b) {
  if (a.n != b.n) {
    throw std::runtime_error("sparse matrix multiply requires square matrices of the same size");
  }

  std::vector<std::map<int, double>> row_maps(static_cast<std::size_t>(a.n));
  for (int row = 0; row < a.n; ++row) {
    std::map<int, double> accum;
    for (int pos_a = a.row_ptr[static_cast<std::size_t>(row)];
         pos_a < a.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos_a) {
      const int mid = a.col_idx[static_cast<std::size_t>(pos_a)];
      const double value_a = a.values[static_cast<std::size_t>(pos_a)];
      for (int pos_b = b.row_ptr[static_cast<std::size_t>(mid)];
           pos_b < b.row_ptr[static_cast<std::size_t>(mid + 1)]; ++pos_b) {
        const int col = b.col_idx[static_cast<std::size_t>(pos_b)];
        accum[col] += value_a * b.values[static_cast<std::size_t>(pos_b)];
      }
    }
    row_maps[static_cast<std::size_t>(row)] = std::move(accum);
  }

  SparseMatrixCSR matrix;
  finalize_row_maps(row_maps, matrix);
  return matrix;
}

inline SparseMatrixCSR build_cahn_hilliard_operator(const SparseMatrixCSR &laplacian, const SparseMatrixCSR &biharmonic,
                                                    double alpha0, double beta1, double beta2) {
  if (laplacian.n != biharmonic.n) {
    throw std::runtime_error("cahn-hilliard operator assembly requires matching matrix sizes");
  }

  std::vector<std::map<int, double>> row_maps(static_cast<std::size_t>(laplacian.n));
  for (int row = 0; row < laplacian.n; ++row) {
    std::map<int, double> accum;
    accum[row] += alpha0;

    for (int pos = laplacian.row_ptr[static_cast<std::size_t>(row)];
         pos < laplacian.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos) {
      const int col = laplacian.col_idx[static_cast<std::size_t>(pos)];
      accum[col] += -beta1 * laplacian.values[static_cast<std::size_t>(pos)];
    }
    for (int pos = biharmonic.row_ptr[static_cast<std::size_t>(row)];
         pos < biharmonic.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos) {
      const int col = biharmonic.col_idx[static_cast<std::size_t>(pos)];
      accum[col] += beta2 * biharmonic.values[static_cast<std::size_t>(pos)];
    }

    row_maps[static_cast<std::size_t>(row)] = std::move(accum);
  }

  SparseMatrixCSR matrix;
  finalize_row_maps(row_maps, matrix);
  return matrix;
}

inline void apply_matrix(const SparseMatrixCSR &matrix, const std::vector<double> &x, std::vector<double> &y) {
  if (static_cast<int>(x.size()) != matrix.n) {
    throw std::runtime_error("matrix-vector multiply size mismatch");
  }

  y.assign(static_cast<std::size_t>(matrix.n), 0.0);
  for (int row = 0; row < matrix.n; ++row) {
    double sum = 0.0;
    for (int pos = matrix.row_ptr[static_cast<std::size_t>(row)];
         pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos) {
      sum += matrix.values[static_cast<std::size_t>(pos)] *
             x[static_cast<std::size_t>(matrix.col_idx[static_cast<std::size_t>(pos)])];
    }
    y[static_cast<std::size_t>(row)] = sum;
  }
}

inline bool try_build_incomplete_ldlt(const SparseMatrixCSR &matrix, KrylovPreconditioner &preconditioner) {
  preconditioner.type = PreconditionerType::incomplete_ldlt;
  preconditioner.lower.assign(matrix.values.size(), 0.0);
  preconditioner.diag.assign(static_cast<std::size_t>(matrix.n), 0.0);
  preconditioner.lower_transpose.assign(static_cast<std::size_t>(matrix.n), {});

  double max_diag = 0.0;
  for (int row = 0; row < matrix.n; ++row) {
    max_diag = std::max(max_diag, std::abs(matrix.values[static_cast<std::size_t>(
                                                matrix.diag_pos[static_cast<std::size_t>(row)])]));
  }
  preconditioner.diagonal_floor = std::max(1.0, max_diag) * 1.0e-12;

  auto find_lower_entry = [&](int row, int target_col) {
    for (int pos = matrix.row_ptr[static_cast<std::size_t>(row)];
         pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos) {
      const int col = matrix.col_idx[static_cast<std::size_t>(pos)];
      if (col >= row || col > target_col) {
        break;
      }
      if (col == target_col) {
        return preconditioner.lower[static_cast<std::size_t>(pos)];
      }
    }
    return 0.0;
  };

  for (int row = 0; row < matrix.n; ++row) {
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
          sum -= preconditioner.lower[static_cast<std::size_t>(prev)] *
                 preconditioner.diag[static_cast<std::size_t>(k)] * l_jk;
        }
      }

      const double denom =
          std::max(preconditioner.diag[static_cast<std::size_t>(col)], preconditioner.diagonal_floor);
      preconditioner.lower[static_cast<std::size_t>(pos)] = sum / denom;
    }

    double d = matrix.values[static_cast<std::size_t>(matrix.diag_pos[static_cast<std::size_t>(row)])];
    for (int pos = matrix.row_ptr[static_cast<std::size_t>(row)];
         pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos) {
      const int col = matrix.col_idx[static_cast<std::size_t>(pos)];
      if (col >= row) {
        break;
      }
      d -= square(preconditioner.lower[static_cast<std::size_t>(pos)]) *
           preconditioner.diag[static_cast<std::size_t>(col)];
    }

    if (!std::isfinite(d) || d <= preconditioner.diagonal_floor) {
      return false;
    }
    preconditioner.diag[static_cast<std::size_t>(row)] = d;
  }

  for (int row = 0; row < matrix.n; ++row) {
    for (int pos = matrix.row_ptr[static_cast<std::size_t>(row)];
         pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos) {
      const int col = matrix.col_idx[static_cast<std::size_t>(pos)];
      if (col >= row) {
        break;
      }
      preconditioner.lower_transpose[static_cast<std::size_t>(col)].push_back(
          {row, preconditioner.lower[static_cast<std::size_t>(pos)]});
    }
  }

  return true;
}

inline bool try_build_incomplete_cholesky(const SparseMatrixCSR &matrix, KrylovPreconditioner &preconditioner) {
  preconditioner.type = PreconditionerType::incomplete_cholesky;
  preconditioner.lower.assign(matrix.values.size(), 0.0);
  preconditioner.diag.assign(static_cast<std::size_t>(matrix.n), 0.0);
  preconditioner.lower_transpose.assign(static_cast<std::size_t>(matrix.n), {});

  double max_diag = 0.0;
  for (int row = 0; row < matrix.n; ++row) {
    max_diag = std::max(max_diag, std::abs(matrix.values[static_cast<std::size_t>(
                                                matrix.diag_pos[static_cast<std::size_t>(row)])]));
  }
  preconditioner.diagonal_floor = std::sqrt(std::max(1.0, max_diag) * 1.0e-12);

  auto find_lower_entry = [&](int row, int target_col) {
    for (int pos = matrix.row_ptr[static_cast<std::size_t>(row)];
         pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos) {
      const int col = matrix.col_idx[static_cast<std::size_t>(pos)];
      if (col >= row || col > target_col) {
        break;
      }
      if (col == target_col) {
        return preconditioner.lower[static_cast<std::size_t>(pos)];
      }
    }
    return 0.0;
  };

  const double diagonal_floor_sq = square(preconditioner.diagonal_floor);
  for (int row = 0; row < matrix.n; ++row) {
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
          sum -= preconditioner.lower[static_cast<std::size_t>(prev)] * l_jk;
        }
      }

      const double denom =
          std::max(preconditioner.diag[static_cast<std::size_t>(col)], preconditioner.diagonal_floor);
      preconditioner.lower[static_cast<std::size_t>(pos)] = sum / denom;
    }

    double d = matrix.values[static_cast<std::size_t>(matrix.diag_pos[static_cast<std::size_t>(row)])];
    for (int pos = matrix.row_ptr[static_cast<std::size_t>(row)];
         pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos) {
      const int col = matrix.col_idx[static_cast<std::size_t>(pos)];
      if (col >= row) {
        break;
      }
      d -= square(preconditioner.lower[static_cast<std::size_t>(pos)]);
    }

    if (!std::isfinite(d) || d <= diagonal_floor_sq) {
      return false;
    }
    preconditioner.diag[static_cast<std::size_t>(row)] = std::sqrt(d);
  }

  for (int row = 0; row < matrix.n; ++row) {
    for (int pos = matrix.row_ptr[static_cast<std::size_t>(row)];
         pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos) {
      const int col = matrix.col_idx[static_cast<std::size_t>(pos)];
      if (col >= row) {
        break;
      }
      preconditioner.lower_transpose[static_cast<std::size_t>(col)].push_back(
          {row, preconditioner.lower[static_cast<std::size_t>(pos)]});
    }
  }

  return true;
}

inline KrylovPreconditioner build_diagonal_preconditioner(const SparseMatrixCSR &matrix) {
  KrylovPreconditioner preconditioner;
  preconditioner.type = PreconditionerType::diagonal;
  preconditioner.diagonal_inverse.resize(static_cast<std::size_t>(matrix.n), 1.0);
  for (int row = 0; row < matrix.n; ++row) {
    const double diagonal =
        matrix.values[static_cast<std::size_t>(matrix.diag_pos[static_cast<std::size_t>(row)])];
    preconditioner.diagonal_inverse[static_cast<std::size_t>(row)] =
        1.0 / std::max(std::abs(diagonal), 1.0e-30);
  }
  return preconditioner;
}

inline KrylovPreconditioner build_preconditioner(const SparseMatrixCSR &matrix) {
  KrylovPreconditioner preconditioner = build_diagonal_preconditioner(matrix);
  KrylovPreconditioner incomplete_ldlt;
  if (try_build_incomplete_ldlt(matrix, incomplete_ldlt)) {
    return incomplete_ldlt;
  }

  return preconditioner;
}

inline void apply_preconditioner(const SparseMatrixCSR &matrix, const KrylovPreconditioner &preconditioner,
                                 const std::vector<double> &r, std::vector<double> &z) {
  if (preconditioner.type == PreconditionerType::diagonal) {
    z.resize(r.size());
    for (std::size_t idx = 0; idx < r.size(); ++idx) {
      z[idx] = preconditioner.diagonal_inverse[idx] * r[idx];
    }
    return;
  }

  if (static_cast<int>(r.size()) != matrix.n) {
    throw std::runtime_error("preconditioner application size mismatch");
  }

  if (preconditioner.type == PreconditionerType::incomplete_cholesky) {
    std::vector<double> y(static_cast<std::size_t>(matrix.n), 0.0);
    for (int row = 0; row < matrix.n; ++row) {
      double sum = r[static_cast<std::size_t>(row)];
      for (int pos = matrix.row_ptr[static_cast<std::size_t>(row)];
           pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos) {
        const int col = matrix.col_idx[static_cast<std::size_t>(pos)];
        if (col >= row) {
          break;
        }
        sum -= preconditioner.lower[static_cast<std::size_t>(pos)] * y[static_cast<std::size_t>(col)];
      }
      y[static_cast<std::size_t>(row)] =
          sum / std::max(preconditioner.diag[static_cast<std::size_t>(row)], preconditioner.diagonal_floor);
    }

    z = y;
    for (int row = matrix.n - 1; row >= 0; --row) {
      double sum = z[static_cast<std::size_t>(row)];
      for (const LowerTransposeEntry &entry : preconditioner.lower_transpose[static_cast<std::size_t>(row)]) {
        sum -= entry.value * z[static_cast<std::size_t>(entry.row)];
      }
      z[static_cast<std::size_t>(row)] =
          sum / std::max(preconditioner.diag[static_cast<std::size_t>(row)], preconditioner.diagonal_floor);
    }
    return;
  }

  std::vector<double> y(static_cast<std::size_t>(matrix.n), 0.0);
  for (int row = 0; row < matrix.n; ++row) {
    double sum = r[static_cast<std::size_t>(row)];
    for (int pos = matrix.row_ptr[static_cast<std::size_t>(row)];
         pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos) {
      const int col = matrix.col_idx[static_cast<std::size_t>(pos)];
      if (col >= row) {
        break;
      }
      sum -= preconditioner.lower[static_cast<std::size_t>(pos)] * y[static_cast<std::size_t>(col)];
    }
    y[static_cast<std::size_t>(row)] = sum;
  }

  for (int row = 0; row < matrix.n; ++row) {
    y[static_cast<std::size_t>(row)] /=
        std::max(preconditioner.diag[static_cast<std::size_t>(row)], preconditioner.diagonal_floor);
  }

  z = y;
  for (int row = matrix.n - 1; row >= 0; --row) {
    double sum = z[static_cast<std::size_t>(row)];
    for (const LowerTransposeEntry &entry : preconditioner.lower_transpose[static_cast<std::size_t>(row)]) {
      sum -= entry.value * z[static_cast<std::size_t>(entry.row)];
    }
    z[static_cast<std::size_t>(row)] = sum;
  }
}

inline LinearSolveReport solve_preconditioned_cg(const SparseMatrixCSR &matrix, const KrylovPreconditioner &preconditioner,
                                                 const std::vector<double> &rhs, int max_iterations, double tolerance,
                                                 std::vector<double> &x,
                                                 const VectorProjection &project_vector = {},
                                                 const CGMonitor &monitor = {}) {
  if (matrix.n <= 0) {
    throw std::runtime_error("cannot solve an empty linear system");
  }
  if (static_cast<int>(rhs.size()) != matrix.n) {
    throw std::runtime_error("pcg rhs size mismatch");
  }

  if (static_cast<int>(x.size()) != matrix.n) {
    x.assign(static_cast<std::size_t>(matrix.n), 0.0);
  }

  std::vector<double> ax;
  apply_matrix(matrix, x, ax);

  std::vector<double> r(static_cast<std::size_t>(matrix.n), 0.0);
  for (int idx = 0; idx < matrix.n; ++idx) {
    r[static_cast<std::size_t>(idx)] = rhs[static_cast<std::size_t>(idx)] - ax[static_cast<std::size_t>(idx)];
  }
  if (project_vector) {
    project_vector(r);
  }

  const double rhs_norm = std::sqrt(std::max(dot_product(rhs, rhs), 0.0));
  LinearSolveReport report;
  report.absolute_residual = std::sqrt(std::max(dot_product(r, r), 0.0));
  report.relative_residual = report.absolute_residual / std::max(rhs_norm, 1.0e-30);
  if (report.relative_residual < tolerance || report.absolute_residual < tolerance) {
    if (monitor) {
      CGMonitorData data;
      data.iteration = 0;
      data.absolute_residual = report.absolute_residual;
      data.relative_residual = report.relative_residual;
      data.iterate_residual = report.iterate_residual;
      data.x = &x;
      data.r = &r;
      monitor(data);
    }
    return report;
  }

  std::vector<double> z;
  apply_preconditioner(matrix, preconditioner, r, z);
  if (project_vector) {
    project_vector(z);
  }
  double rz_old = dot_product(r, z);
  if (!std::isfinite(rz_old) || rz_old <= 0.0) {
    throw std::runtime_error("pcg preconditioner breakdown: non-positive r^T M^{-1} r");
  }
  if (monitor) {
    CGMonitorData data;
    data.iteration = 0;
    data.absolute_residual = report.absolute_residual;
    data.relative_residual = report.relative_residual;
    data.iterate_residual = report.iterate_residual;
    data.preconditioned_residual = std::sqrt(std::max(rz_old, 0.0));
    data.x = &x;
    data.r = &r;
    data.z = &z;
    monitor(data);
  }

  std::vector<double> p = z;
  std::vector<double> ap(static_cast<std::size_t>(matrix.n), 0.0);
  const int iteration_limit = std::max(1, max_iterations);
  for (int iter = 0; iter < iteration_limit; ++iter) {
    apply_matrix(matrix, p, ap);
    const double pap = dot_product(p, ap);
    if (!std::isfinite(pap) || std::abs(pap) < 1.0e-30) {
      throw std::runtime_error("pcg breakdown: p^T A p is invalid");
    }

    const double alpha = rz_old / pap;
    double update_sq = 0.0;
    double x_norm_sq = 0.0;
    for (int idx = 0; idx < matrix.n; ++idx) {
      const double delta = alpha * p[static_cast<std::size_t>(idx)];
      x[static_cast<std::size_t>(idx)] += delta;
      r[static_cast<std::size_t>(idx)] -= alpha * ap[static_cast<std::size_t>(idx)];
      update_sq += delta * delta;
      x_norm_sq += x[static_cast<std::size_t>(idx)] * x[static_cast<std::size_t>(idx)];
    }
    if (project_vector) {
      project_vector(x);
      project_vector(r);
    }

    report.iterations = iter + 1;
    report.iterate_residual = std::sqrt(update_sq / std::max(x_norm_sq, 1.0e-30));
    report.absolute_residual = std::sqrt(std::max(dot_product(r, r), 0.0));
    report.relative_residual = report.absolute_residual / std::max(rhs_norm, 1.0e-30);
    if (report.relative_residual < tolerance || report.absolute_residual < tolerance) {
      if (monitor) {
        CGMonitorData data;
        data.iteration = iter + 1;
        data.alpha = alpha;
        data.absolute_residual = report.absolute_residual;
        data.relative_residual = report.relative_residual;
        data.iterate_residual = report.iterate_residual;
        data.x = &x;
        data.r = &r;
        monitor(data);
      }
      break;
    }
    double beta = 0.0;
    apply_preconditioner(matrix, preconditioner, r, z);
    if (project_vector) {
      project_vector(z);
    }
    const double rz_new = dot_product(r, z);
    if (!std::isfinite(rz_new) || rz_new <= 0.0) {
      throw std::runtime_error("pcg preconditioner breakdown: non-positive new r^T M^{-1} r");
    }

    beta = rz_new / rz_old;
    if (monitor) {
      CGMonitorData data;
      data.iteration = iter + 1;
      data.alpha = alpha;
      data.beta = beta;
      data.absolute_residual = report.absolute_residual;
      data.relative_residual = report.relative_residual;
      data.iterate_residual = report.iterate_residual;
      data.preconditioned_residual = std::sqrt(std::max(rz_new, 0.0));
      data.x = &x;
      data.r = &r;
      data.z = &z;
      monitor(data);
    }
    for (int idx = 0; idx < matrix.n; ++idx) {
      p[static_cast<std::size_t>(idx)] = z[static_cast<std::size_t>(idx)] + beta * p[static_cast<std::size_t>(idx)];
    }
    if (project_vector) {
      project_vector(p);
    }
    rz_old = rz_new;
  }

  return report;
}

inline LinearSolveReport solve_dcdm_conjugate_directions(const SparseMatrixCSR &matrix, const std::vector<double> &rhs,
                                                         int max_iterations, double tolerance, std::vector<double> &x,
                                                         const DirectionGenerator &direction_generator,
                                                         const DCDMOptions &options = {},
                                                         const VectorProjection &project_vector = {},
                                                         const CGMonitor &monitor = {}) {
  if (matrix.n <= 0) {
    throw std::runtime_error("cannot solve an empty linear system");
  }
  if (static_cast<int>(rhs.size()) != matrix.n) {
    throw std::runtime_error("dcdm rhs size mismatch");
  }

  if (static_cast<int>(x.size()) != matrix.n) {
    x.assign(static_cast<std::size_t>(matrix.n), 0.0);
  }

  std::vector<double> ax;
  apply_matrix(matrix, x, ax);

  std::vector<double> r(static_cast<std::size_t>(matrix.n), 0.0);
  for (int idx = 0; idx < matrix.n; ++idx) {
    r[static_cast<std::size_t>(idx)] = rhs[static_cast<std::size_t>(idx)] - ax[static_cast<std::size_t>(idx)];
  }
  if (project_vector) {
    project_vector(r);
    project_vector(x);
  }

  const double rhs_norm = std::sqrt(std::max(dot_product(rhs, rhs), 0.0));
  LinearSolveReport report;
  report.absolute_residual = l2_norm(r);
  report.relative_residual = report.absolute_residual / std::max(rhs_norm, 1.0e-30);
  if (monitor) {
    CGMonitorData data;
    data.iteration = 0;
    data.absolute_residual = report.absolute_residual;
    data.relative_residual = report.relative_residual;
    data.iterate_residual = report.iterate_residual;
    data.x = &x;
    data.r = &r;
    monitor(data);
  }
  if (report.relative_residual < tolerance || report.absolute_residual < tolerance) {
    return report;
  }

  std::vector<std::vector<double>> directions;
  std::vector<std::vector<double>> adirections;
  const int iteration_limit = std::max(1, max_iterations);
  const int max_keep = std::max(options.max_stored_directions, std::max(options.history_size, 1));

  for (int iter = 0; iter < iteration_limit; ++iter) {
    if (options.restart_interval > 0 && iter > 0 && (iter % options.restart_interval) == 0) {
      directions.clear();
      adirections.clear();
    }

    const double residual_norm = l2_norm(r);
    if (residual_norm < tolerance) {
      break;
    }

    std::vector<double> r_hat = r;
    for (double &value : r_hat) {
      value /= std::max(residual_norm, options.direction_floor);
    }

    std::vector<double> z;
    const DirectionGenerationContext context{iter, residual_norm, &x, &r, &rhs};
    bool direction_ok = direction_generator && direction_generator(r_hat, z, context);
    if (!direction_ok || static_cast<int>(z.size()) != matrix.n || !vector_is_finite(z) ||
        l2_norm(z) < options.direction_floor) {
      z = r;
    }
    if (project_vector) {
      project_vector(z);
    }
    if (!vector_is_finite(z) || l2_norm(z) < options.direction_floor) {
      z = r;
      if (project_vector) {
        project_vector(z);
      }
    }

    std::vector<double> d = z;
    int history_begin = 0;
    if (options.history_size > 0 && static_cast<int>(directions.size()) > options.history_size) {
      history_begin = static_cast<int>(directions.size()) - options.history_size;
    }
    for (int idx = history_begin; idx < static_cast<int>(directions.size()); ++idx) {
      const double denom_hist = dot_product(directions[static_cast<std::size_t>(idx)],
                                            adirections[static_cast<std::size_t>(idx)]);
      if (!std::isfinite(denom_hist) || std::abs(denom_hist) < options.direction_floor) {
        continue;
      }
      const double coeff = dot_product(d, adirections[static_cast<std::size_t>(idx)]) / denom_hist;
      for (int j = 0; j < matrix.n; ++j) {
        d[static_cast<std::size_t>(j)] -= coeff * directions[static_cast<std::size_t>(idx)][static_cast<std::size_t>(j)];
      }
    }
    if (project_vector) {
      project_vector(d);
    }

    std::vector<double> ad;
    apply_matrix(matrix, d, ad);
    double rd = dot_product(r, d);
    double dad = dot_product(d, ad);
    if (!vector_is_finite(d) || !vector_is_finite(ad) || l2_norm(d) < options.direction_floor ||
        !std::isfinite(rd) || rd <= 0.0 || !std::isfinite(dad) || std::abs(dad) < options.direction_floor) {
      d = r;
      if (project_vector) {
        project_vector(d);
      }
      apply_matrix(matrix, d, ad);
      rd = dot_product(r, d);
      dad = dot_product(d, ad);
      if (!std::isfinite(rd) || rd <= 0.0 || !std::isfinite(dad) || std::abs(dad) < options.direction_floor) {
        throw std::runtime_error("dcdm breakdown: invalid fallback direction");
      }
    }

    const double alpha = rd / dad;
    double update_sq = 0.0;
    for (int idx = 0; idx < matrix.n; ++idx) {
      const double delta = alpha * d[static_cast<std::size_t>(idx)];
      x[static_cast<std::size_t>(idx)] += delta;
      update_sq += delta * delta;
    }
    if (project_vector) {
      project_vector(x);
    }

    apply_matrix(matrix, x, ax);
    for (int idx = 0; idx < matrix.n; ++idx) {
      r[static_cast<std::size_t>(idx)] = rhs[static_cast<std::size_t>(idx)] - ax[static_cast<std::size_t>(idx)];
    }
    if (project_vector) {
      project_vector(r);
    }

    double x_norm_sq = dot_product(x, x);
    report.iterations = iter + 1;
    report.iterate_residual = std::sqrt(update_sq / std::max(x_norm_sq, 1.0e-30));
    report.absolute_residual = l2_norm(r);
    report.relative_residual = report.absolute_residual / std::max(rhs_norm, 1.0e-30);

    if (monitor) {
      CGMonitorData data;
      data.iteration = iter + 1;
      data.alpha = alpha;
      data.absolute_residual = report.absolute_residual;
      data.relative_residual = report.relative_residual;
      data.iterate_residual = report.iterate_residual;
      data.preconditioned_residual = l2_norm(z);
      data.x = &x;
      data.r = &r;
      data.z = &z;
      monitor(data);
    }

    directions.push_back(d);
    adirections.push_back(ad);
    if (static_cast<int>(directions.size()) > max_keep) {
      directions.erase(directions.begin());
      adirections.erase(adirections.begin());
    }

    if (report.relative_residual < tolerance || report.absolute_residual < tolerance) {
      break;
    }
  }

  return report;
}

inline LinearSolveReport solve_preconditioned_bicgstab(const SparseMatrixCSR &matrix,
                                                       const KrylovPreconditioner &preconditioner,
                                                       const std::vector<double> &rhs, int max_iterations,
                                                       double tolerance, std::vector<double> &x,
                                                       bool use_relative_tolerance = true) {
  if (matrix.n <= 0) {
    throw std::runtime_error("cannot solve an empty linear system");
  }
  if (static_cast<int>(rhs.size()) != matrix.n) {
    throw std::runtime_error("bicgstab rhs size mismatch");
  }

  if (static_cast<int>(x.size()) != matrix.n) {
    x.assign(static_cast<std::size_t>(matrix.n), 0.0);
  }

  std::vector<double> ax;
  apply_matrix(matrix, x, ax);

  std::vector<double> r(static_cast<std::size_t>(matrix.n), 0.0);
  for (int idx = 0; idx < matrix.n; ++idx) {
    r[static_cast<std::size_t>(idx)] = rhs[static_cast<std::size_t>(idx)] - ax[static_cast<std::size_t>(idx)];
  }

  const double rhs_norm = std::sqrt(std::max(dot_product(rhs, rhs), 0.0));
  LinearSolveReport report;
  report.absolute_residual = std::sqrt(std::max(dot_product(r, r), 0.0));
  report.relative_residual = report.absolute_residual / std::max(rhs_norm, 1.0e-30);
  if ((use_relative_tolerance && report.relative_residual < tolerance) || report.absolute_residual < tolerance) {
    return report;
  }

  std::vector<double> r_hat = r;
  std::vector<double> p(static_cast<std::size_t>(matrix.n), 0.0);
  std::vector<double> v(static_cast<std::size_t>(matrix.n), 0.0);
  std::vector<double> phat;
  std::vector<double> s(static_cast<std::size_t>(matrix.n), 0.0);
  std::vector<double> shat;
  std::vector<double> t(static_cast<std::size_t>(matrix.n), 0.0);
  std::vector<double> update(static_cast<std::size_t>(matrix.n), 0.0);

  double rho_old = 1.0;
  double alpha = 1.0;
  double omega = 1.0;
  const int iteration_limit = std::max(1, max_iterations);

  for (int iter = 0; iter < iteration_limit; ++iter) {
    const double rho_new = dot_product(r_hat, r);
    if (!std::isfinite(rho_new) || std::abs(rho_new) < 1.0e-30) {
      throw std::runtime_error("bicgstab breakdown: invalid rho");
    }

    const double beta = (iter == 0) ? 0.0 : (rho_new / rho_old) * (alpha / omega);
    for (int idx = 0; idx < matrix.n; ++idx) {
      p[static_cast<std::size_t>(idx)] =
          r[static_cast<std::size_t>(idx)] +
          beta * (p[static_cast<std::size_t>(idx)] - omega * v[static_cast<std::size_t>(idx)]);
    }

    apply_preconditioner(matrix, preconditioner, p, phat);
    apply_matrix(matrix, phat, v);
    const double rhat_v = dot_product(r_hat, v);
    if (!std::isfinite(rhat_v) || std::abs(rhat_v) < 1.0e-30) {
      throw std::runtime_error("bicgstab breakdown: invalid r_hat^T A M^{-1} p");
    }

    alpha = rho_new / rhat_v;
    for (int idx = 0; idx < matrix.n; ++idx) {
      s[static_cast<std::size_t>(idx)] =
          r[static_cast<std::size_t>(idx)] - alpha * v[static_cast<std::size_t>(idx)];
      update[static_cast<std::size_t>(idx)] = alpha * phat[static_cast<std::size_t>(idx)];
    }

    double s_norm = std::sqrt(std::max(dot_product(s, s), 0.0));
    if ((use_relative_tolerance && s_norm / std::max(rhs_norm, 1.0e-30) < tolerance) || s_norm < tolerance) {
      double update_sq = 0.0;
      double x_norm_sq = 0.0;
      for (int idx = 0; idx < matrix.n; ++idx) {
        x[static_cast<std::size_t>(idx)] += update[static_cast<std::size_t>(idx)];
        update_sq += update[static_cast<std::size_t>(idx)] * update[static_cast<std::size_t>(idx)];
        x_norm_sq += x[static_cast<std::size_t>(idx)] * x[static_cast<std::size_t>(idx)];
      }
      report.iterations = iter + 1;
      report.iterate_residual = std::sqrt(update_sq / std::max(x_norm_sq, 1.0e-30));
      report.absolute_residual = s_norm;
      report.relative_residual = s_norm / std::max(rhs_norm, 1.0e-30);
      return report;
    }

    apply_preconditioner(matrix, preconditioner, s, shat);
    apply_matrix(matrix, shat, t);
    const double tt = dot_product(t, t);
    if (!std::isfinite(tt) || std::abs(tt) < 1.0e-30) {
      throw std::runtime_error("bicgstab breakdown: invalid t^T t");
    }

    omega = dot_product(t, s) / tt;
    if (!std::isfinite(omega) || std::abs(omega) < 1.0e-30) {
      throw std::runtime_error("bicgstab breakdown: invalid omega");
    }

    double update_sq = 0.0;
    double x_norm_sq = 0.0;
    for (int idx = 0; idx < matrix.n; ++idx) {
      update[static_cast<std::size_t>(idx)] += omega * shat[static_cast<std::size_t>(idx)];
      x[static_cast<std::size_t>(idx)] += update[static_cast<std::size_t>(idx)];
      r[static_cast<std::size_t>(idx)] =
          s[static_cast<std::size_t>(idx)] - omega * t[static_cast<std::size_t>(idx)];
      update_sq += update[static_cast<std::size_t>(idx)] * update[static_cast<std::size_t>(idx)];
      x_norm_sq += x[static_cast<std::size_t>(idx)] * x[static_cast<std::size_t>(idx)];
    }

    report.iterations = iter + 1;
    report.iterate_residual = std::sqrt(update_sq / std::max(x_norm_sq, 1.0e-30));
    report.absolute_residual = std::sqrt(std::max(dot_product(r, r), 0.0));
    report.relative_residual = report.absolute_residual / std::max(rhs_norm, 1.0e-30);
    if ((use_relative_tolerance && report.relative_residual < tolerance) || report.absolute_residual < tolerance) {
      return report;
    }

    rho_old = rho_new;
  }

  return report;
}

} // namespace ding::ch_sparse_krylov
