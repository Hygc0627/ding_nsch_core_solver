#include "solver.hpp"
#include "core/linear_algebra/ch_sparse_krylov.hpp"

#include <algorithm>
#include <cctype>
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
#include <sys/wait.h>
#include <unistd.h>
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
                                               const KrylovPreconditioner &preconditioner, std::vector<double> &x,
                                               const ch_sparse_krylov::VectorProjection &project_vector = {}) {
  x.assign(static_cast<std::size_t>(matrix.n), 0.0);
  return ch_sparse_krylov::solve_preconditioned_cg(matrix, preconditioner, rhs, max_iterations, tolerance, x,
                                                   project_vector);
}

std::string shell_quote(const std::string &value) {
  std::string quoted = "'";
  for (char ch : value) {
    if (ch == '\'') {
      quoted += "'\\''";
    } else {
      quoted += ch;
    }
  }
  quoted += "'";
  return quoted;
}

void write_pressure_vector_file(const std::filesystem::path &path, const std::vector<double> &values) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("cannot open pressure vector file: " + path.string());
  }
  out << values.size() << "\n";
  out << std::setprecision(17);
  for (double value : values) {
    out << value << "\n";
  }
}

std::vector<double> read_pressure_vector_file(const std::filesystem::path &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open pressure vector file: " + path.string());
  }

  int n = 0;
  in >> n;
  if (n < 0) {
    throw std::runtime_error("invalid pressure vector length in file: " + path.string());
  }

  std::vector<double> values(static_cast<std::size_t>(n), 0.0);
  for (int idx = 0; idx < n; ++idx) {
    if (!(in >> values[static_cast<std::size_t>(idx)])) {
      throw std::runtime_error("failed reading pressure vector file: " + path.string());
    }
  }
  return values;
}

void scale_pressure_linear_system(std::vector<std::map<int, double>> &row_maps, std::vector<double> &rhs,
                                  double scale_factor) {
  if (scale_factor == 1.0) {
    return;
  }
  for (auto &row : row_maps) {
    for (auto &[col, value] : row) {
      (void)col;
      value *= scale_factor;
    }
  }
  for (double &value : rhs) {
    value *= scale_factor;
  }
}

} // namespace

void Solver::build_pressure_linear_system(std::vector<std::map<int, double>> &row_maps, std::vector<double> &rhs,
                                          double &max_diag, bool use_split, double rho_ref,
                                          const Field2D *pressure_extrapolated) const {
  const int n = cfg_.nx * cfg_.ny;
  row_maps.assign(static_cast<std::size_t>(n), {});
  rhs.assign(static_cast<std::size_t>(n), 0.0);
  max_diag = 0.0;

  auto row_index = [&](int i, int j) { return i * cfg_.ny + j; };

  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      const int row = row_index(i, j);
      const bool has_east = cfg_.periodic_x || i < cfg_.nx - 1;
      const bool has_west = cfg_.periodic_x || i > 0;
      const bool has_north = cfg_.periodic_y || j < cfg_.ny - 1;
      const bool has_south = cfg_.periodic_y || j > 0;

      const double coeff_e = use_split ? 1.0 / (dx_ * dx_) : 1.0 / (rho_u_face(rho_mid_, i + 1, j) * dx_ * dx_);
      const double coeff_w = use_split ? 1.0 / (dx_ * dx_) : 1.0 / (rho_u_face(rho_mid_, i, j) * dx_ * dx_);
      const double coeff_n = use_split ? 1.0 / (dy_ * dy_) : 1.0 / (rho_v_face(rho_mid_, i, j + 1) * dy_ * dy_);
      const double coeff_s = use_split ? 1.0 / (dy_ * dy_) : 1.0 / (rho_v_face(rho_mid_, i, j) * dy_ * dy_);

      double diag = 0.0;
      double rhs_value =
          use_split ? -(liu_split_explicit_divergence(*pressure_extrapolated, i, j) +
                        rho_ref * divergence_cell(u_star_, v_star_, i, j) / cfg_.dt)
                    : -divergence_cell(u_star_, v_star_, i, j) / cfg_.dt;

      auto add_boundary = [&](BoundarySide side, double coeff, double spacing) {
        const BoundaryConditionSpec bc = effective_pressure_bc(side);
        if (bc.type == BoundaryConditionType::dirichlet) {
          diag += 2.0 * coeff;
          rhs_value += 2.0 * coeff * bc.value;
        } else {
          rhs_value += coeff * bc.value * spacing;
        }
      };

      if (has_east) {
        diag += coeff_e;
        const int ie = cfg_.periodic_x && i == cfg_.nx - 1 ? 0 : i + 1;
        row_maps[static_cast<std::size_t>(row)][row_index(ie, j)] += -coeff_e;
      } else {
        add_boundary(BoundarySide::right, coeff_e, dx_);
      }
      if (has_west) {
        diag += coeff_w;
        const int iw = cfg_.periodic_x && i == 0 ? cfg_.nx - 1 : i - 1;
        row_maps[static_cast<std::size_t>(row)][row_index(iw, j)] += -coeff_w;
      } else {
        add_boundary(BoundarySide::left, coeff_w, dx_);
      }
      if (has_north) {
        diag += coeff_n;
        const int jn = cfg_.periodic_y && j == cfg_.ny - 1 ? 0 : j + 1;
        row_maps[static_cast<std::size_t>(row)][row_index(i, jn)] += -coeff_n;
      } else {
        add_boundary(BoundarySide::top, coeff_n, dy_);
      }
      if (has_south) {
        diag += coeff_s;
        const int js = cfg_.periodic_y && j == 0 ? cfg_.ny - 1 : j - 1;
        row_maps[static_cast<std::size_t>(row)][row_index(i, js)] += -coeff_s;
      } else {
        add_boundary(BoundarySide::bottom, coeff_s, dy_);
      }

      row_maps[static_cast<std::size_t>(row)][row] += diag;
      rhs[static_cast<std::size_t>(row)] = rhs_value;
      max_diag = std::max(max_diag, diag);
    }
  }
}

void Solver::regularize_pressure_linear_system(std::vector<std::map<int, double>> &row_maps, std::vector<double> &rhs,
                                               double max_diag) const {
  (void)row_maps;
  (void)max_diag;
  if (pressure_has_dirichlet_boundary()) {
    return;
  }

  const int n = cfg_.nx * cfg_.ny;
  const double rhs_mean =
      std::accumulate(rhs.begin(), rhs.end(), 0.0) / std::max(static_cast<double>(n), 1.0);
  for (double &value : rhs) {
    value -= rhs_mean;
  }
}

PressureLinearSystem Solver::assemble_pressure_linear_system(bool use_split, double rho_ref,
                                                             const Field2D *pressure_extrapolated) const {
  PressureLinearSystem system;
  system.use_split = use_split;
  system.rho_ref = rho_ref;
  system.has_dirichlet_boundary = pressure_has_dirichlet_boundary();
  build_pressure_linear_system(system.row_maps, system.rhs, system.max_diag, use_split, rho_ref, pressure_extrapolated);
  regularize_pressure_linear_system(system.row_maps, system.rhs, system.max_diag);
  return system;
}

ch_sparse_krylov::VectorProjection Solver::pressure_nullspace_projection() const {
  if (pressure_has_dirichlet_boundary()) {
    return {};
  }
  return [this](std::vector<double> &values) {
    if (values.empty()) {
      return;
    }
    const double mean =
        std::accumulate(values.begin(), values.end(), 0.0) / std::max(static_cast<double>(values.size()), 1.0);
    for (double &value : values) {
      value -= mean;
    }
  };
}

void Solver::scatter_pressure_solution(const std::vector<double> &x) {
  auto row_index = [&](int i, int j) { return i * cfg_.ny + j; };
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      pressure_correction_(i, j) = x[static_cast<std::size_t>(row_index(i, j))];
    }
  }
}

double Solver::solve_pressure_correction_jacobi() {
  pressure_correction_.fill(0.0);
  apply_pressure_bc(pressure_correction_);
  last_pressure_iterations_ = 0;
  std::vector<std::map<int, double>> row_maps;
  std::vector<double> rhs;
  double max_diag = 0.0;
  build_pressure_linear_system(row_maps, rhs, max_diag, false, 1.0, nullptr);
  regularize_pressure_linear_system(row_maps, rhs, max_diag);
  const SparseMatrixCSR matrix = finalize_pressure_matrix(row_maps);
  std::vector<double> x(static_cast<std::size_t>(matrix.n), 0.0);
  std::vector<double> x_new(static_cast<std::size_t>(matrix.n), 0.0);

  double residual = 0.0;
  const double omega = 1.0;
  for (int iter = 0; iter < cfg_.poisson_iterations; ++iter) {
    double max_residual = 0.0;
    for (int row = 0; row < matrix.n; ++row) {
      double diag = 0.0;
      double ax = 0.0;
      for (int pos = matrix.row_ptr[static_cast<std::size_t>(row)];
           pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos) {
        const int col = matrix.col_idx[static_cast<std::size_t>(pos)];
        const double value = matrix.values[static_cast<std::size_t>(pos)];
        ax += value * x[static_cast<std::size_t>(col)];
        if (col == row) {
          diag = value;
        }
      }
      const double row_residual = rhs[static_cast<std::size_t>(row)] - ax;
      if (!std::isfinite(row_residual) || !std::isfinite(diag) || std::abs(diag) < 1.0e-30) {
        throw std::runtime_error("Jacobi pressure solve diverged");
      }
      x_new[static_cast<std::size_t>(row)] =
          x[static_cast<std::size_t>(row)] + omega * row_residual / std::max(std::abs(diag), 1.0e-30);
      max_residual = std::max(max_residual, std::abs(row_residual));
    }
    x.swap(x_new);
    residual = max_residual;
    last_pressure_iterations_ = iter + 1;
    if (residual < cfg_.pressure_tolerance) {
      break;
    }
  }
  scatter_pressure_solution(x);
  if (!pressure_has_dirichlet_boundary()) {
    subtract_mean(pressure_correction_);
  }
  apply_pressure_bc(pressure_correction_);
  return residual;
}

double Solver::solve_pressure_correction_icpcg() {
  pressure_correction_.fill(0.0);
  apply_pressure_bc(pressure_correction_);
  last_pressure_iterations_ = 0;
  std::vector<std::map<int, double>> row_maps;
  std::vector<double> rhs;
  const PressureLinearSystem system = assemble_pressure_linear_system(false, 1.0, nullptr);
  const SparseMatrixCSR matrix = finalize_pressure_matrix(system.row_maps);
  std::vector<double> x(static_cast<std::size_t>(matrix.n), 0.0);
  const KrylovPreconditioner preconditioner = build_icc_preconditioner_strict(matrix);
  const LinearSolveReport report = solve_pressure_system_strict(
      matrix, system.rhs, cfg_.poisson_iterations, cfg_.pressure_tolerance, preconditioner, x, pressure_nullspace_projection());
  last_pressure_iterations_ = report.iterations;
  scatter_pressure_solution(x);
  if (!pressure_has_dirichlet_boundary()) {
    subtract_mean(pressure_correction_);
  }
  apply_pressure_bc(pressure_correction_);
  return report.relative_residual;
}

double Solver::solve_pressure_correction_ildlt_pcg() {
  pressure_correction_.fill(0.0);
  apply_pressure_bc(pressure_correction_);
  last_pressure_iterations_ = 0;
  std::vector<std::map<int, double>> row_maps;
  std::vector<double> rhs;
  const PressureLinearSystem system = assemble_pressure_linear_system(false, 1.0, nullptr);
  const SparseMatrixCSR matrix = finalize_pressure_matrix(system.row_maps);
  std::vector<double> x(static_cast<std::size_t>(matrix.n), 0.0);
  const KrylovPreconditioner preconditioner = build_ildlt_preconditioner_strict(matrix);
  const LinearSolveReport report = solve_pressure_system_strict(
      matrix, system.rhs, cfg_.poisson_iterations, cfg_.pressure_tolerance, preconditioner, x, pressure_nullspace_projection());
  last_pressure_iterations_ = report.iterations;
  scatter_pressure_solution(x);
  if (!pressure_has_dirichlet_boundary()) {
    subtract_mean(pressure_correction_);
  }
  apply_pressure_bc(pressure_correction_);
  return report.relative_residual;
}

double Solver::solve_pressure_correction_liu_split_icpcg() {
  pressure_correction_.fill(0.0);
  apply_pressure_bc(pressure_correction_);
  last_pressure_iterations_ = 0;

  Field2D pressure_extrapolated(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0);
  build_liu_split_pressure_extrapolation(pressure_extrapolated);
  const double rho_ref = liu_split_reference_density();
  const PressureLinearSystem system = assemble_pressure_linear_system(true, rho_ref, &pressure_extrapolated);
  const SparseMatrixCSR matrix = finalize_pressure_matrix(system.row_maps);
  std::vector<double> x(static_cast<std::size_t>(matrix.n), 0.0);
  const KrylovPreconditioner preconditioner = build_icc_preconditioner_strict(matrix);
  const LinearSolveReport report = solve_pressure_system_strict(
      matrix, system.rhs, cfg_.poisson_iterations, cfg_.pressure_tolerance, preconditioner, x, pressure_nullspace_projection());
  last_pressure_iterations_ = report.iterations;
  scatter_pressure_solution(x);
  if (!pressure_has_dirichlet_boundary()) {
    subtract_mean(pressure_correction_);
  }
  apply_pressure_bc(pressure_correction_);
  return report.relative_residual;
}

double Solver::solve_pressure_correction_liu_split_dcdm_icpcg() {
  namespace fs = std::filesystem;

  pressure_correction_.fill(0.0);
  apply_pressure_bc(pressure_correction_);
  last_pressure_iterations_ = 0;

  Field2D pressure_extrapolated(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0);
  build_liu_split_pressure_extrapolation(pressure_extrapolated);
  const double rho_ref = liu_split_reference_density();
  const PressureLinearSystem system = assemble_pressure_linear_system(true, rho_ref, &pressure_extrapolated);
  const SparseMatrixCSR matrix = finalize_pressure_matrix(system.row_maps);
  std::vector<double> x(static_cast<std::size_t>(matrix.n), 0.0);
  const KrylovPreconditioner preconditioner = build_icc_preconditioner_strict(matrix);
  const ch_sparse_krylov::VectorProjection project_vector = pressure_nullspace_projection();

  std::string direction_mode = cfg_.dcdm_direction_mode;
  std::transform(direction_mode.begin(), direction_mode.end(), direction_mode.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });

  const bool use_identity_direction = direction_mode == "identity";
  const bool use_icc_direction = direction_mode == "icc" || direction_mode == "icc_preconditioned" ||
                                 direction_mode == "pcg_preconditioned";
  const bool use_neural_direction =
      direction_mode == "neural" || direction_mode == "hydea_neural" || direction_mode == "ml";
  if (!use_identity_direction && !use_icc_direction && !use_neural_direction) {
    throw std::runtime_error("unsupported dcdm_direction_mode: " + cfg_.dcdm_direction_mode);
  }

  const fs::path solver_dir = pressure_solver_dir();
  fs::create_directories(solver_dir);
  const fs::path direction_input_path = solver_dir / "dcdm_direction_input.txt";
  const fs::path direction_output_path = solver_dir / "dcdm_direction_output.txt";

  bool neural_direction_enabled = use_neural_direction;
  bool neural_direction_failed = false;
  if (use_neural_direction) {
    if (cfg_.hydea_model_path.empty()) {
      throw std::runtime_error("dcdm neural direction requires hydea_model_path");
    }
    const fs::path options_path = cfg_.hydea_solver_config;
    if (!fs::exists(options_path)) {
      throw std::runtime_error("HyDEA direction options file not found: " + options_path.string());
    }
    start_dcdm_direction_worker(options_path.string(), cfg_.hydea_model_path, cfg_.nx, cfg_.ny);
  }

  const auto icc_direction = [&](const std::vector<double> &residual_like, std::vector<double> &candidate) {
    ch_sparse_krylov::apply_preconditioner(matrix, preconditioner, residual_like, candidate);
    return true;
  };

  const ch_sparse_krylov::DirectionGenerator direction_generator =
      [&](const std::vector<double> &normalized_residual, std::vector<double> &candidate_direction,
          const ch_sparse_krylov::DirectionGenerationContext &context) {
        (void)context;
        if (use_identity_direction) {
          candidate_direction = normalized_residual;
          return true;
        }
        if (use_icc_direction) {
          return icc_direction(normalized_residual, candidate_direction);
        }
        if (!neural_direction_enabled) {
          return icc_direction(normalized_residual, candidate_direction);
        }

        try {
          write_pressure_vector_file(direction_input_path, normalized_residual);
          request_dcdm_direction_via_worker(direction_input_path.string(), direction_output_path.string());
          candidate_direction = read_pressure_vector_file(direction_output_path);
          if (static_cast<int>(candidate_direction.size()) != matrix.n) {
            throw std::runtime_error("neural direction size mismatch");
          }
          return true;
        } catch (const std::exception &ex) {
          neural_direction_enabled = false;
          if (!neural_direction_failed) {
            neural_direction_failed = true;
            log_message("WARN step=" + std::to_string(current_step_index_ + 1) +
                        " dcdm_neural_direction_fallback=icc reason=" + ex.what());
          }
          return icc_direction(normalized_residual, candidate_direction);
        }
      };

  ch_sparse_krylov::DCDMOptions options;
  options.history_size = cfg_.dcdm_history_size;
  options.max_stored_directions = cfg_.dcdm_max_stored_directions;
  options.restart_interval = cfg_.dcdm_restart_interval;
  const LinearSolveReport report = ch_sparse_krylov::solve_dcdm_conjugate_directions(
      matrix, system.rhs, cfg_.poisson_iterations, cfg_.pressure_tolerance, x, direction_generator, options,
      project_vector);
  last_pressure_iterations_ = report.iterations;
  scatter_pressure_solution(x);
  if (!pressure_has_dirichlet_boundary()) {
    subtract_mean(pressure_correction_);
  }
  apply_pressure_bc(pressure_correction_);
  return report.relative_residual;
}

double Solver::solve_pressure_correction_liu_split_ildlt_pcg() {
  pressure_correction_.fill(0.0);
  apply_pressure_bc(pressure_correction_);
  last_pressure_iterations_ = 0;

  Field2D pressure_extrapolated(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0);
  build_liu_split_pressure_extrapolation(pressure_extrapolated);
  const double rho_ref = liu_split_reference_density();
  const PressureLinearSystem system = assemble_pressure_linear_system(true, rho_ref, &pressure_extrapolated);
  const SparseMatrixCSR matrix = finalize_pressure_matrix(system.row_maps);
  std::vector<double> x(static_cast<std::size_t>(matrix.n), 0.0);
  const KrylovPreconditioner preconditioner = build_ildlt_preconditioner_strict(matrix);
  const LinearSolveReport report = solve_pressure_system_strict(
      matrix, system.rhs, cfg_.poisson_iterations, cfg_.pressure_tolerance, preconditioner, x, pressure_nullspace_projection());
  last_pressure_iterations_ = report.iterations;
  scatter_pressure_solution(x);
  if (!pressure_has_dirichlet_boundary()) {
    subtract_mean(pressure_correction_);
  }
  apply_pressure_bc(pressure_correction_);
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
  std::ostringstream rhs_name;
  rhs_name << "rhs_step_" << std::setw(6) << std::setfill('0') << (current_step_index_ + 1) << ".txt";
  const fs::path rhs_step_path = solver_dir / rhs_name.str();

  const bool use_split = cfg_.pressure_scheme == "split_petsc_pcg" ||
                         cfg_.pressure_scheme == "paper_split_petsc_pcg" ||
                         cfg_.pressure_scheme == "liu_split_petsc_pcg";
  std::vector<std::map<int, double>> row_maps;
  std::vector<double> rhs_values;
  double max_diag = 0.0;
  Field2D pressure_extrapolated(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0);
  double rho_ref = 1.0;
  if (use_split) {
    build_liu_split_pressure_extrapolation(pressure_extrapolated);
    rho_ref = liu_split_reference_density();
  }
  build_pressure_linear_system(row_maps, rhs_values, max_diag, use_split, rho_ref,
                               use_split ? &pressure_extrapolated : nullptr);
  regularize_pressure_linear_system(row_maps, rhs_values, max_diag);
  const SparseMatrixCSR matrix = finalize_pressure_matrix(row_maps);
  const int n = matrix.n;
  const std::size_t nnz = matrix.values.size();
  const bool matrix_is_reusable = can_reuse_petsc_pressure_matrix(use_split);

  if (!matrix_is_reusable || !petsc_pressure_worker_in_ || petsc_pressure_worker_matrix_path_ != matrix_path.string() ||
      petsc_pressure_worker_options_path_ != cfg_.petsc_solver_config ||
      petsc_pressure_worker_use_constant_nullspace_ != !pressure_has_dirichlet_boundary()) {
    std::ofstream matrix_out(matrix_path);
    if (!matrix_out) {
      throw std::runtime_error("cannot open PETSc pressure matrix file: " + matrix_path.string());
    }
    matrix_out << n << " " << n << " " << nnz << "\n";
    matrix_out << std::setprecision(17);
    for (int row = 0; row < matrix.n; ++row) {
      for (int pos = matrix.row_ptr[static_cast<std::size_t>(row)];
           pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos) {
        matrix_out << row << " " << matrix.col_idx[static_cast<std::size_t>(pos)] << " "
                   << matrix.values[static_cast<std::size_t>(pos)] << "\n";
      }
    }
  }

  write_pressure_vector_file(rhs_path, rhs_values);
  if (write_monitor_log) {
    write_pressure_vector_file(rhs_step_path, rhs_values);
  }

  const fs::path options_path = cfg_.petsc_solver_config;
  if (!fs::exists(options_path)) {
    throw std::runtime_error("PETSc pressure options file not found: " + options_path.string());
  }
  const std::string log_prefix =
      cfg_.verbose ? "[pressure step " + std::to_string(current_step_index_ + 1) + " outer " +
                         std::to_string(current_coupling_iteration_ + 1) + "]"
                   : "";
  if (matrix_is_reusable) {
    if (!petsc_pressure_worker_in_) {
      start_petsc_pressure_worker(matrix_path.string(), options_path.string(), !pressure_has_dirichlet_boundary());
    }
    solve_pressure_correction_petsc_via_worker(rhs_path.string(), solution_path.string(), report_path.string(),
                                               write_monitor_log ? monitor_log_path.string() : "", log_prefix);
  } else {
    const fs::path script_path = cfg_.petsc_solver_script;
    if (!fs::exists(script_path)) {
      throw std::runtime_error("PETSc pressure helper script not found: " + script_path.string());
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
    command << shell_quote(cfg_.petsc_python_executable) << " " << shell_quote(script_path.string()) << " --matrix "
            << shell_quote(matrix_path.string()) << " --rhs " << shell_quote(rhs_path.string()) << " --solution "
            << shell_quote(solution_path.string()) << " --report " << shell_quote(report_path.string()) << " --config "
            << shell_quote(options_path.string()) << " --use-constant-nullspace "
            << (pressure_has_dirichlet_boundary() ? "false" : "true");
    if (!log_prefix.empty()) {
      command << " --log-prefix " << shell_quote(log_prefix);
    }
    if (write_monitor_log) {
      command << " --monitor-log " << shell_quote(monitor_log_path.string());
    }

    const int code = std::system(command.str().c_str());
    if (code != 0) {
      throw std::runtime_error("PETSc pressure solve failed with exit code " + std::to_string(code));
    }
  }

  std::vector<double> x(static_cast<std::size_t>(n), 0.0);
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
    for (int idx = 0; idx < n; ++idx) {
      solution_in >> x[static_cast<std::size_t>(idx)];
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

  scatter_pressure_solution(x);
  if (!pressure_has_dirichlet_boundary()) {
    subtract_mean(pressure_correction_);
  }
  apply_pressure_bc(pressure_correction_);
  return residual;
}

bool Solver::can_reuse_petsc_pressure_matrix(bool use_split) const {
  return use_split || std::abs(cfg_.density_ratio - 1.0) < 1.0e-12;
}

void Solver::start_petsc_pressure_worker(const std::string &matrix_path, const std::string &options_path,
                                         bool use_constant_nullspace) {
  namespace fs = std::filesystem;
  stop_petsc_pressure_worker();

  const fs::path script_path = cfg_.petsc_solver_script;
  if (!fs::exists(script_path)) {
    throw std::runtime_error("PETSc pressure helper script not found: " + script_path.string());
  }

  int stdin_pipe[2] = {-1, -1};
  int stdout_pipe[2] = {-1, -1};
  if (pipe(stdin_pipe) != 0 || pipe(stdout_pipe) != 0) {
    throw std::runtime_error("failed to create pipes for PETSc pressure worker");
  }

  const pid_t pid = fork();
  if (pid < 0) {
    close(stdin_pipe[0]);
    close(stdin_pipe[1]);
    close(stdout_pipe[0]);
    close(stdout_pipe[1]);
    throw std::runtime_error("failed to fork PETSc pressure worker");
  }

  if (pid == 0) {
    dup2(stdin_pipe[0], STDIN_FILENO);
    dup2(stdout_pipe[1], STDOUT_FILENO);
    close(stdin_pipe[0]);
    close(stdin_pipe[1]);
    close(stdout_pipe[0]);
    close(stdout_pipe[1]);
    execlp(cfg_.petsc_python_executable.c_str(), cfg_.petsc_python_executable.c_str(), script_path.string().c_str(),
           "--server", "--matrix", matrix_path.c_str(), "--config", options_path.c_str(),
           "--use-constant-nullspace", use_constant_nullspace ? "true" : "false", static_cast<char *>(nullptr));
    _exit(127);
  }

  close(stdin_pipe[0]);
  close(stdout_pipe[1]);
  petsc_pressure_worker_pid_ = pid;
  petsc_pressure_worker_in_ = fdopen(stdin_pipe[1], "w");
  petsc_pressure_worker_out_ = fdopen(stdout_pipe[0], "r");
  if (!petsc_pressure_worker_in_ || !petsc_pressure_worker_out_) {
    stop_petsc_pressure_worker();
    throw std::runtime_error("failed to open PETSc pressure worker streams");
  }
  petsc_pressure_worker_matrix_path_ = matrix_path;
  petsc_pressure_worker_options_path_ = options_path;
  petsc_pressure_worker_use_constant_nullspace_ = use_constant_nullspace;
}

void Solver::stop_petsc_pressure_worker() {
  if (petsc_pressure_worker_in_ && petsc_pressure_worker_out_) {
    std::fputs("EXIT\n", petsc_pressure_worker_in_);
    std::fflush(petsc_pressure_worker_in_);
    char buffer[4096];
    char *response = std::fgets(buffer, sizeof(buffer), petsc_pressure_worker_out_);
    (void)response;
  }
  if (petsc_pressure_worker_in_) {
    std::fclose(petsc_pressure_worker_in_);
    petsc_pressure_worker_in_ = nullptr;
  }
  if (petsc_pressure_worker_out_) {
    std::fclose(petsc_pressure_worker_out_);
    petsc_pressure_worker_out_ = nullptr;
  }
  if (petsc_pressure_worker_pid_ > 0) {
    int status = 0;
    waitpid(petsc_pressure_worker_pid_, &status, 0);
    petsc_pressure_worker_pid_ = -1;
  }
  petsc_pressure_worker_matrix_path_.clear();
  petsc_pressure_worker_options_path_.clear();
  petsc_pressure_worker_use_constant_nullspace_ = false;
}

void Solver::start_dcdm_direction_worker(const std::string &options_path, const std::string &model_path, int grid_nx,
                                         int grid_ny) {
  namespace fs = std::filesystem;

  if (dcdm_direction_worker_in_ && dcdm_direction_worker_out_ && dcdm_direction_worker_options_path_ == options_path &&
      dcdm_direction_worker_model_path_ == model_path && dcdm_direction_worker_grid_nx_ == grid_nx &&
      dcdm_direction_worker_grid_ny_ == grid_ny) {
    return;
  }
  stop_dcdm_direction_worker();

  const fs::path script_path = cfg_.dcdm_direction_script;
  if (!fs::exists(script_path)) {
    throw std::runtime_error("DCDM direction helper script not found: " + script_path.string());
  }
  if (!fs::exists(fs::path(options_path))) {
    throw std::runtime_error("DCDM direction options file not found: " + options_path);
  }

  int stdin_pipe[2] = {-1, -1};
  int stdout_pipe[2] = {-1, -1};
  if (pipe(stdin_pipe) != 0 || pipe(stdout_pipe) != 0) {
    throw std::runtime_error("failed to create pipes for DCDM direction worker");
  }

  const pid_t pid = fork();
  if (pid < 0) {
    close(stdin_pipe[0]);
    close(stdin_pipe[1]);
    close(stdout_pipe[0]);
    close(stdout_pipe[1]);
    throw std::runtime_error("failed to fork DCDM direction worker");
  }

  if (pid == 0) {
    const std::string nx_text = std::to_string(grid_nx);
    const std::string ny_text = std::to_string(grid_ny);
    dup2(stdin_pipe[0], STDIN_FILENO);
    dup2(stdout_pipe[1], STDOUT_FILENO);
    close(stdin_pipe[0]);
    close(stdin_pipe[1]);
    close(stdout_pipe[0]);
    close(stdout_pipe[1]);
    execlp(cfg_.petsc_python_executable.c_str(), cfg_.petsc_python_executable.c_str(), script_path.string().c_str(),
           "--server", "--config", options_path.c_str(), "--model-path", model_path.c_str(), "--grid-nx",
           nx_text.c_str(), "--grid-ny", ny_text.c_str(), static_cast<char *>(nullptr));
    _exit(127);
  }

  close(stdin_pipe[0]);
  close(stdout_pipe[1]);
  dcdm_direction_worker_pid_ = pid;
  dcdm_direction_worker_in_ = fdopen(stdin_pipe[1], "w");
  dcdm_direction_worker_out_ = fdopen(stdout_pipe[0], "r");
  if (!dcdm_direction_worker_in_ || !dcdm_direction_worker_out_) {
    stop_dcdm_direction_worker();
    throw std::runtime_error("failed to open DCDM direction worker streams");
  }
  dcdm_direction_worker_options_path_ = options_path;
  dcdm_direction_worker_model_path_ = model_path;
  dcdm_direction_worker_grid_nx_ = grid_nx;
  dcdm_direction_worker_grid_ny_ = grid_ny;
}

void Solver::stop_dcdm_direction_worker() {
  if (dcdm_direction_worker_in_ && dcdm_direction_worker_out_) {
    std::fputs("EXIT\n", dcdm_direction_worker_in_);
    std::fflush(dcdm_direction_worker_in_);
    char buffer[4096];
    char *response = std::fgets(buffer, sizeof(buffer), dcdm_direction_worker_out_);
    (void)response;
  }
  if (dcdm_direction_worker_in_) {
    std::fclose(dcdm_direction_worker_in_);
    dcdm_direction_worker_in_ = nullptr;
  }
  if (dcdm_direction_worker_out_) {
    std::fclose(dcdm_direction_worker_out_);
    dcdm_direction_worker_out_ = nullptr;
  }
  if (dcdm_direction_worker_pid_ > 0) {
    int status = 0;
    waitpid(dcdm_direction_worker_pid_, &status, 0);
    dcdm_direction_worker_pid_ = -1;
  }
  dcdm_direction_worker_options_path_.clear();
  dcdm_direction_worker_model_path_.clear();
  dcdm_direction_worker_grid_nx_ = 0;
  dcdm_direction_worker_grid_ny_ = 0;
}

void Solver::request_dcdm_direction_via_worker(const std::string &input_path, const std::string &output_path) {
  if (!dcdm_direction_worker_in_ || !dcdm_direction_worker_out_) {
    throw std::runtime_error("DCDM direction worker is not active");
  }

  std::fputs("DIRECTION\n", dcdm_direction_worker_in_);
  std::fputs((input_path + "\n").c_str(), dcdm_direction_worker_in_);
  std::fputs((output_path + "\n").c_str(), dcdm_direction_worker_in_);
  std::fflush(dcdm_direction_worker_in_);

  char buffer[4096];
  if (!std::fgets(buffer, sizeof(buffer), dcdm_direction_worker_out_)) {
    stop_dcdm_direction_worker();
    throw std::runtime_error("DCDM direction worker terminated unexpectedly");
  }
  std::string response(buffer);
  if (response.rfind("OK", 0) == 0) {
    return;
  }
  stop_dcdm_direction_worker();
  if (response.rfind("ERROR ", 0) == 0) {
    throw std::runtime_error("DCDM direction worker failed: " + response.substr(6));
  }
  throw std::runtime_error("DCDM direction worker returned malformed response");
}

void Solver::solve_pressure_correction_petsc_via_worker(const std::string &rhs_path, const std::string &solution_path,
                                                        const std::string &report_path,
                                                        const std::string &monitor_log_path,
                                                        const std::string &log_prefix) {
  if (!petsc_pressure_worker_in_ || !petsc_pressure_worker_out_) {
    throw std::runtime_error("PETSc pressure worker is not active");
  }
  std::fputs("SOLVE\n", petsc_pressure_worker_in_);
  std::fputs((rhs_path + "\n").c_str(), petsc_pressure_worker_in_);
  std::fputs((solution_path + "\n").c_str(), petsc_pressure_worker_in_);
  std::fputs((report_path + "\n").c_str(), petsc_pressure_worker_in_);
  std::fputs((monitor_log_path + "\n").c_str(), petsc_pressure_worker_in_);
  std::fputs((log_prefix + "\n").c_str(), petsc_pressure_worker_in_);
  std::fflush(petsc_pressure_worker_in_);

  char buffer[4096];
  if (!std::fgets(buffer, sizeof(buffer), petsc_pressure_worker_out_)) {
    stop_petsc_pressure_worker();
    throw std::runtime_error("PETSc pressure worker terminated unexpectedly");
  }
  std::string response(buffer);
  if (response.rfind("OK", 0) == 0) {
    return;
  }
  stop_petsc_pressure_worker();
  if (response.rfind("ERROR ", 0) == 0) {
    throw std::runtime_error("PETSc pressure worker failed: " + response.substr(6));
  }
  throw std::runtime_error("PETSc pressure worker returned malformed response");
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
  std::ostringstream rhs_name;
  rhs_name << "rhs_step_" << std::setw(6) << std::setfill('0') << (current_step_index_ + 1) << ".txt";
  const fs::path rhs_step_path = solver_dir / rhs_name.str();

  std::vector<std::map<int, double>> row_maps;
  std::vector<double> rhs_values;
  double max_diag = 0.0;
  const bool use_split = cfg_.pressure_scheme == "split_hydea" || cfg_.pressure_scheme == "paper_split_hydea" ||
                         cfg_.pressure_scheme == "liu_split_hydea";
  Field2D pressure_extrapolated(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0);
  double rho_ref = 1.0;
  if (use_split) {
    build_liu_split_pressure_extrapolation(pressure_extrapolated);
    rho_ref = liu_split_reference_density();
  }
  build_pressure_linear_system(row_maps, rhs_values, max_diag, use_split, rho_ref,
                               use_split ? &pressure_extrapolated : nullptr);
  if (use_split && cfg_.hydea_scale_split_system) {
    // Match the local HyDEA training operator, which uses the normalized
    // five-point Laplacian with O(1) coefficients (diag in [2, 4], offdiag -1).
    if (std::abs(dx_ - dy_) > 1.0e-12) {
      throw std::runtime_error(
          "split HyDEA currently requires dx == dy so the Poisson operator can be normalized");
    }
    const double h2 = dx_ * dx_;
    scale_pressure_linear_system(row_maps, rhs_values, h2);
    max_diag *= h2;
  }
  regularize_pressure_linear_system(row_maps, rhs_values, max_diag);
  const SparseMatrixCSR matrix = finalize_pressure_matrix(row_maps);
  const int n = matrix.n;
  const std::size_t nnz = matrix.values.size();

  {
    std::ofstream matrix_out(matrix_path);
    if (!matrix_out) {
      throw std::runtime_error("cannot open HyDEA pressure matrix file: " + matrix_path.string());
    }
    matrix_out << n << " " << n << " " << nnz << "\n";
    matrix_out << std::setprecision(17);
    for (int row = 0; row < matrix.n; ++row) {
      for (int pos = matrix.row_ptr[static_cast<std::size_t>(row)];
           pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos) {
        matrix_out << row << " " << matrix.col_idx[static_cast<std::size_t>(pos)] << " "
                   << matrix.values[static_cast<std::size_t>(pos)] << "\n";
      }
    }
  }

  write_pressure_vector_file(rhs_path, rhs_values);
  if (write_monitor_log) {
    write_pressure_vector_file(rhs_step_path, rhs_values);
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
  command << shell_quote(cfg_.petsc_python_executable) << " " << shell_quote(script_path.string())
          << " --matrix " << shell_quote(matrix_path.string())
          << " --rhs " << shell_quote(rhs_path.string())
          << " --solution " << shell_quote(solution_path.string())
          << " --report " << shell_quote(report_path.string())
          << " --config " << shell_quote(options_path.string())
          << " --model-path " << shell_quote(cfg_.hydea_model_path)
          << " --grid-nx " << cfg_.nx
          << " --grid-ny " << cfg_.ny;
  if (cfg_.verbose) {
    command << " --log-prefix \"[pressure step " << (current_step_index_ + 1) << " outer "
            << (current_coupling_iteration_ + 1) << "]\"";
  }
  if (write_monitor_log) {
    command << " --monitor-log " << shell_quote(monitor_log_path.string());
  }

  const int code = std::system(command.str().c_str());
  if (code != 0) {
    throw std::runtime_error("HyDEA pressure solve failed with exit code " + std::to_string(code));
  }

  std::vector<double> x(static_cast<std::size_t>(n), 0.0);
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
    for (int idx = 0; idx < n; ++idx) {
      solution_in >> x[static_cast<std::size_t>(idx)];
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

  scatter_pressure_solution(x);
  if (!pressure_has_dirichlet_boundary()) {
    subtract_mean(pressure_correction_);
  }
  apply_pressure_bc(pressure_correction_);
  return residual;
}

double Solver::solve_pressure_correction() {
  const char *solver_name = nullptr;
  double residual = 0.0;
  if (cfg_.pressure_scheme == "petsc_pcg") {
    solver_name = "PETSc";
    residual = solve_pressure_correction_petsc();
  } else if (cfg_.pressure_scheme == "liu_split_petsc_pcg" || cfg_.pressure_scheme == "split_petsc_pcg" ||
             cfg_.pressure_scheme == "paper_split_petsc_pcg") {
    solver_name = "SplitPETSc";
    residual = solve_pressure_correction_petsc();
  } else if (cfg_.pressure_scheme == "liu_split_hydea" || cfg_.pressure_scheme == "split_hydea" ||
             cfg_.pressure_scheme == "paper_split_hydea") {
    solver_name = "SplitHyDEA";
    residual = solve_pressure_correction_hydea();
  } else if (cfg_.pressure_scheme == "hydea") {
    solver_name = "HyDEA";
    residual = solve_pressure_correction_hydea();
  } else if (cfg_.pressure_scheme == "liu_split_icpcg" || cfg_.pressure_scheme == "split_icpcg" ||
             cfg_.pressure_scheme == "paper_split_icpcg") {
    solver_name = "SplitICCPCG";
    residual = solve_pressure_correction_liu_split_icpcg();
  } else if (cfg_.pressure_scheme == "liu_split_dcdm_icpcg" || cfg_.pressure_scheme == "split_dcdm_icpcg" ||
             cfg_.pressure_scheme == "paper_split_dcdm_icpcg") {
    solver_name = "SplitDCDMICC";
    residual = solve_pressure_correction_liu_split_dcdm_icpcg();
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
  if (analysis_mode_enabled("online") &&
      (cfg_.pressure_scheme == "icpcg" || cfg_.pressure_scheme == "ildlt_pcg" ||
       cfg_.pressure_scheme == "liu_split_icpcg" || cfg_.pressure_scheme == "split_icpcg" ||
       cfg_.pressure_scheme == "paper_split_icpcg" || cfg_.pressure_scheme == "liu_split_dcdm_icpcg" ||
       cfg_.pressure_scheme == "split_dcdm_icpcg" || cfg_.pressure_scheme == "paper_split_dcdm_icpcg" ||
       cfg_.pressure_scheme == "liu_split_ildlt_pcg" ||
       cfg_.pressure_scheme == "split_ildlt_pcg" || cfg_.pressure_scheme == "paper_split_ildlt_pcg") &&
      current_step_index_ + 1 == cfg_.analysis_trigger_step) {
    run_pressure_analysis_from_snapshot(make_pressure_analysis_snapshot("online"));
  }
  return residual;
}


} // namespace ding
