#include "solver.hpp"
#include "core/linear_algebra/ch_sparse_krylov.hpp"

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

namespace {

using ch_sparse_krylov::LinearSolveReport;
using ch_sparse_krylov::KrylovPreconditioner;
using ch_sparse_krylov::SparseMatrixCSR;

SparseMatrixCSR finalize_pressure_matrix(const std::vector<std::map<int, double>> &row_maps) {
  SparseMatrixCSR matrix;
  ch_sparse_krylov::finalize_row_maps(row_maps, matrix);
  return matrix;
}

KrylovPreconditioner build_icc_preconditioner_strict(const SparseMatrixCSR &matrix) {
  KrylovPreconditioner preconditioner;
  if (!ch_sparse_krylov::try_build_incomplete_cholesky(matrix, preconditioner)) {
    throw std::runtime_error("ICC preconditioner construction failed");
  }
  return preconditioner;
}

KrylovPreconditioner build_ildlt_preconditioner_strict(const SparseMatrixCSR &matrix) {
  KrylovPreconditioner preconditioner;
  if (!ch_sparse_krylov::try_build_incomplete_ldlt(matrix, preconditioner)) {
    throw std::runtime_error("ILDLT preconditioner construction failed");
  }
  return preconditioner;
}

LinearSolveReport solve_pressure_system_strict(const SparseMatrixCSR &matrix, const std::vector<double> &rhs,
                                               int max_iterations, double tolerance,
                                               const KrylovPreconditioner &preconditioner, std::vector<double> &x) {
  x.assign(static_cast<std::size_t>(matrix.n), 0.0);
  return ch_sparse_krylov::solve_preconditioned_cg(matrix, preconditioner, rhs, max_iterations, tolerance, x);
}

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

  const SparseMatrixCSR matrix = finalize_pressure_matrix(row_maps);
  std::vector<double> x(static_cast<std::size_t>(n), 0.0);
  const KrylovPreconditioner preconditioner = build_icc_preconditioner_strict(matrix);
  const LinearSolveReport report =
      solve_pressure_system_strict(matrix, rhs, cfg_.poisson_iterations, cfg_.pressure_tolerance, preconditioner, x);
  last_pressure_iterations_ = report.iterations;

  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      pressure_correction_(i, j) = x[static_cast<std::size_t>(row_index(i, j))];
    }
  }
  subtract_mean(pressure_correction_);
  apply_scalar_bc(pressure_correction_);
  return report.relative_residual;
}

double Solver::solve_pressure_correction_ildlt_pcg() {
  pressure_correction_.fill(0.0);
  apply_scalar_bc(pressure_correction_);
  last_pressure_iterations_ = 0;

  const int n = cfg_.nx * cfg_.ny;
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

  const SparseMatrixCSR matrix = finalize_pressure_matrix(row_maps);
  std::vector<double> x(static_cast<std::size_t>(n), 0.0);
  const KrylovPreconditioner preconditioner = build_ildlt_preconditioner_strict(matrix);
  const LinearSolveReport report =
      solve_pressure_system_strict(matrix, rhs, cfg_.poisson_iterations, cfg_.pressure_tolerance, preconditioner, x);
  last_pressure_iterations_ = report.iterations;

  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      pressure_correction_(i, j) = x[static_cast<std::size_t>(row_index(i, j))];
    }
  }
  subtract_mean(pressure_correction_);
  apply_scalar_bc(pressure_correction_);
  return report.relative_residual;
}

double Solver::solve_pressure_correction_liu_split_icpcg() {
  pressure_correction_.fill(0.0);
  apply_scalar_bc(pressure_correction_);
  last_pressure_iterations_ = 0;

  Field2D pressure_extrapolated(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0);
  build_liu_split_pressure_extrapolation(pressure_extrapolated);
  const double rho_ref = liu_split_reference_density();

  const int n = cfg_.nx * cfg_.ny;
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

  const SparseMatrixCSR matrix = finalize_pressure_matrix(row_maps);
  std::vector<double> x(static_cast<std::size_t>(n), 0.0);
  const KrylovPreconditioner preconditioner = build_icc_preconditioner_strict(matrix);
  const LinearSolveReport report =
      solve_pressure_system_strict(matrix, rhs, cfg_.poisson_iterations, cfg_.pressure_tolerance, preconditioner, x);
  last_pressure_iterations_ = report.iterations;

  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      pressure_correction_(i, j) = x[static_cast<std::size_t>(row_index(i, j))];
    }
  }
  subtract_mean(pressure_correction_);
  apply_scalar_bc(pressure_correction_);
  return report.relative_residual;
}

double Solver::solve_pressure_correction_liu_split_ildlt_pcg() {
  pressure_correction_.fill(0.0);
  apply_scalar_bc(pressure_correction_);
  last_pressure_iterations_ = 0;

  Field2D pressure_extrapolated(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0);
  build_liu_split_pressure_extrapolation(pressure_extrapolated);
  const double rho_ref = liu_split_reference_density();

  const int n = cfg_.nx * cfg_.ny;
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

  const SparseMatrixCSR matrix = finalize_pressure_matrix(row_maps);
  std::vector<double> x(static_cast<std::size_t>(n), 0.0);
  const KrylovPreconditioner preconditioner = build_ildlt_preconditioner_strict(matrix);
  const LinearSolveReport report =
      solve_pressure_system_strict(matrix, rhs, cfg_.poisson_iterations, cfg_.pressure_tolerance, preconditioner, x);
  last_pressure_iterations_ = report.iterations;

  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      pressure_correction_(i, j) = x[static_cast<std::size_t>(row_index(i, j))];
    }
  }
  subtract_mean(pressure_correction_);
  apply_scalar_bc(pressure_correction_);
  return report.relative_residual;
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
  const char *solver_name = nullptr;
  double residual = 0.0;
  if (cfg_.pressure_scheme == "petsc_pcg") {
    solver_name = "PETSc";
    residual = solve_pressure_correction_petsc();
  } else if (cfg_.pressure_scheme == "hydea") {
    solver_name = "HyDEA";
    residual = solve_pressure_correction_hydea();
  } else if (cfg_.pressure_scheme == "liu_split_icpcg" || cfg_.pressure_scheme == "split_icpcg" ||
             cfg_.pressure_scheme == "paper_split_icpcg") {
    solver_name = "SplitICCPCG";
    residual = solve_pressure_correction_liu_split_icpcg();
  } else if (cfg_.pressure_scheme == "liu_split_ildlt_pcg" || cfg_.pressure_scheme == "split_ildlt_pcg" ||
             cfg_.pressure_scheme == "paper_split_ildlt_pcg") {
    solver_name = "SplitILDLTPCG";
    residual = solve_pressure_correction_liu_split_ildlt_pcg();
  } else if (cfg_.pressure_scheme == "icpcg") {
    solver_name = "ICCPCG";
    residual = solve_pressure_correction_icpcg();
  } else if (cfg_.pressure_scheme == "ildlt_pcg") {
    solver_name = "ILDLTPCG";
    residual = solve_pressure_correction_ildlt_pcg();
  } else if (cfg_.pressure_scheme == "jacobi") {
    solver_name = "Jacobi";
    residual = solve_pressure_correction_jacobi();
  } else {
    throw std::runtime_error("unsupported pressure scheme: " + cfg_.pressure_scheme);
  }
  last_pressure_solver_name_ = solver_name;
  if (cfg_.verbose) {
    std::cout << "[pressure step " << (current_step_index_ + 1) << " outer " << (current_coupling_iteration_ + 1)
              << "] [" << solver_name << "] done iterations=" << last_pressure_iterations_ << " residual="
              << std::scientific << std::setprecision(6) << residual << "\n";
  }
  return residual;
}


} // namespace ding
