#include "solver.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace ding {

namespace {

using ch_sparse_krylov::CGMonitorData;
using ch_sparse_krylov::KrylovPreconditioner;
using ch_sparse_krylov::LinearSolveReport;
using ch_sparse_krylov::PreconditionerType;
using ch_sparse_krylov::SparseMatrixCSR;

SparseMatrixCSR finalize_pressure_matrix(const std::vector<std::map<int, double>> &row_maps) {
  SparseMatrixCSR matrix;
  ch_sparse_krylov::finalize_row_maps(row_maps, matrix);
  return matrix;
}

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

std::vector<double> field_from_vector(const std::vector<double> &values, int nx, int ny) {
  std::vector<double> field(static_cast<std::size_t>(nx * ny), 0.0);
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      field[static_cast<std::size_t>(j * nx + i)] = values[static_cast<std::size_t>(i * ny + j)];
    }
  }
  return field;
}

void write_matrix_csv(const std::filesystem::path &path, const std::vector<double> &values, int nx, int ny) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("cannot open pressure analysis field file: " + path.string());
  }
  out << std::setprecision(17);
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      if (i > 0) {
        out << ",";
      }
      out << values[static_cast<std::size_t>(j * nx + i)];
    }
    out << "\n";
  }
}

double vector_mean(const std::vector<double> &values) {
  if (values.empty()) {
    return 0.0;
  }
  double sum = 0.0;
  for (double value : values) {
    sum += value;
  }
  return sum / static_cast<double>(values.size());
}

double compute_effective_sigma(const Config &cfg) {
  if (cfg.surface_tension_multiplier == 0.0) {
    return 0.0;
  }
  return cfg.surface_tension_multiplier / std::max(cfg.re * cfg.ca, 1.0e-30);
}

const char *preconditioner_name(PreconditionerType type) {
  if (type == PreconditionerType::incomplete_cholesky) {
    return "ICC";
  }
  if (type == PreconditionerType::incomplete_ldlt) {
    return "ILDLT";
  }
  return "Diagonal";
}

void append_summary_row(const std::filesystem::path &path, const std::string &row) {
  const bool exists = std::filesystem::exists(path);
  std::ofstream out(path, std::ios::app);
  if (!out) {
    throw std::runtime_error("cannot open pressure analysis summary file: " + path.string());
  }
  if (!exists) {
    out << "case_name,case_group,source_mode,step,time,grid_size,density_ratio,sigma,viscosity_ratio,solver_name,"
           "preconditioner_name,max_iter,tolerance,initial_guess_type,rhs_volume_mean,rhs_sum,"
           "pressure_mean_before_fix,pressure_mean_after_fix,nullspace_treatment,final_iter,final_true_res_l2,"
           "final_rel_true_res_l2\n";
  }
  out << row << "\n";
}

} // namespace

std::vector<int> Solver::analysis_key_iterations() const {
  std::vector<int> values;
  std::stringstream in(cfg_.analysis_spectrum_iterations);
  std::string token;
  while (std::getline(in, token, ',')) {
    token.erase(std::remove_if(token.begin(), token.end(), [](unsigned char ch) { return std::isspace(ch); }),
                token.end());
    if (token.empty() || lower_copy(token) == "final") {
      continue;
    }
    values.push_back(std::stoi(token));
  }
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}

bool Solver::analysis_mode_enabled(const std::string &mode_name) const {
  if (!cfg_.analysis_enabled) {
    return false;
  }
  const std::string mode = lower_copy(cfg_.analysis_mode);
  const std::string query = lower_copy(mode_name);
  if (mode == "both") {
    return query == "online" || query == "frozen";
  }
  return mode == query;
}

std::string Solver::pressure_analysis_dir() const {
  namespace fs = std::filesystem;
  return (fs::path(case_output_dir()) / "pressure_analysis").string();
}

PressureAnalysisSnapshot Solver::make_pressure_analysis_snapshot(const std::string &source_mode) const {
  PressureAnalysisSnapshot snapshot(cfg_.nx, cfg_.ny, cfg_.ghost);
  snapshot.phase = c_;
  snapshot.rho_mid = rho_mid_;
  snapshot.u_star = u_star_;
  snapshot.v_star = v_star_;
  snapshot.pressure = pressure_;
  snapshot.pressure_previous = pressure_previous_step_;
  snapshot.step = current_step_index_ + 1;
  snapshot.time = static_cast<double>(snapshot.step) * cfg_.dt;
  snapshot.source_mode = source_mode;
  return snapshot;
}

void Solver::run_pressure_analysis_from_snapshot(const PressureAnalysisSnapshot &snapshot) const {
  namespace fs = std::filesystem;

  const bool use_split = cfg_.pressure_scheme == "liu_split_icpcg" || cfg_.pressure_scheme == "split_icpcg" ||
                         cfg_.pressure_scheme == "paper_split_icpcg" || cfg_.pressure_scheme == "liu_split_dcdm_icpcg" ||
                         cfg_.pressure_scheme == "split_dcdm_icpcg" ||
                         cfg_.pressure_scheme == "paper_split_dcdm_icpcg" ||
                         cfg_.pressure_scheme == "liu_split_ildlt_pcg" ||
                         cfg_.pressure_scheme == "split_ildlt_pcg" || cfg_.pressure_scheme == "paper_split_ildlt_pcg";
  Field2D pressure_extrapolated(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0);
  const Field2D *pressure_extrapolated_ptr = nullptr;
  double rho_ref = 1.0;
  if (use_split) {
    build_liu_split_pressure_extrapolation(pressure_extrapolated);
    pressure_extrapolated_ptr = &pressure_extrapolated;
    rho_ref = liu_split_reference_density();
  }

  const PressureLinearSystem system = assemble_pressure_linear_system(use_split, rho_ref, pressure_extrapolated_ptr);
  const SparseMatrixCSR matrix = finalize_pressure_matrix(system.row_maps);
  const ch_sparse_krylov::VectorProjection project_vector = pressure_nullspace_projection();

  KrylovPreconditioner preconditioner;
  std::string solver_name = "ICCPCG";
  if (cfg_.pressure_scheme == "ildlt_pcg" || cfg_.pressure_scheme == "liu_split_ildlt_pcg" ||
      cfg_.pressure_scheme == "split_ildlt_pcg" || cfg_.pressure_scheme == "paper_split_ildlt_pcg") {
    solver_name = use_split ? "SplitILDLTPCG" : "ILDLTPCG";
    if (!ch_sparse_krylov::try_build_incomplete_ldlt(matrix, preconditioner)) {
      preconditioner = ch_sparse_krylov::build_diagonal_preconditioner(matrix);
    }
  } else {
    solver_name =
        (cfg_.pressure_scheme == "liu_split_dcdm_icpcg" || cfg_.pressure_scheme == "split_dcdm_icpcg" ||
         cfg_.pressure_scheme == "paper_split_dcdm_icpcg")
            ? "SplitDCDMICC"
            : (use_split ? "SplitICCPCG" : "ICCPCG");
    if (!ch_sparse_krylov::try_build_incomplete_cholesky(matrix, preconditioner)) {
      preconditioner = ch_sparse_krylov::build_diagonal_preconditioner(matrix);
    }
  }

  fs::create_directories(pressure_analysis_dir());
  const std::ostringstream stem_builder = [&]() {
    std::ostringstream stem;
    stem << cfg_.name << "_" << snapshot.source_mode << "_step_" << std::setw(6) << std::setfill('0') << snapshot.step;
    return stem;
  }();
  const std::string stem = stem_builder.str();
  const fs::path iter_path = fs::path(pressure_analysis_dir()) / (stem + "_pcg.csv");
  const fs::path field_dir = fs::path(pressure_analysis_dir()) / (stem + "_fields");
  fs::create_directories(field_dir);

  std::ofstream iter_out(iter_path);
  if (!iter_out) {
    throw std::runtime_error("cannot open pressure analysis iteration file: " + iter_path.string());
  }
  iter_out << "case_name,case_group,source_mode,step,time,grid_size,density_ratio,sigma,viscosity_ratio,solver_name,"
              "preconditioner_name,max_iter,tolerance,initial_guess_type,iter,true_res_l2,rel_true_res_l2,"
              "true_res_linf,precond_res,alpha_k,beta_k,q_k,interface_res_l2,bulk_res_l2,interface_res_fraction,"
              "bulk_res_fraction\n";
  iter_out << std::setprecision(17);

  std::vector<double> interface_mask(static_cast<std::size_t>(cfg_.nx * cfg_.ny), 0.0);
  const double phi_cutoff =
      std::tanh(cfg_.analysis_interface_band_multiplier / (2.0 * std::sqrt(2.0)));
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      const double phi = 2.0 * snapshot.phase(i, j) - 1.0;
      interface_mask[static_cast<std::size_t>(i * cfg_.ny + j)] = (std::abs(phi) < phi_cutoff) ? 1.0 : 0.0;
    }
  }

  std::vector<double> rhs_field = field_from_vector(system.rhs, cfg_.nx, cfg_.ny);
  write_matrix_csv(field_dir / "rhs.csv", rhs_field, cfg_.nx, cfg_.ny);
  write_matrix_csv(field_dir / "interface_mask.csv", field_from_vector(interface_mask, cfg_.nx, cfg_.ny), cfg_.nx,
                   cfg_.ny);

  std::vector<double> rho_field(static_cast<std::size_t>(cfg_.nx * cfg_.ny), 0.0);
  std::vector<double> phase_field(static_cast<std::size_t>(cfg_.nx * cfg_.ny), 0.0);
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      rho_field[static_cast<std::size_t>(j * cfg_.nx + i)] = snapshot.rho_mid(i, j);
      phase_field[static_cast<std::size_t>(j * cfg_.nx + i)] = 2.0 * snapshot.phase(i, j) - 1.0;
    }
  }
  write_matrix_csv(field_dir / "rho_mid.csv", rho_field, cfg_.nx, cfg_.ny);
  write_matrix_csv(field_dir / "phi.csv", phase_field, cfg_.nx, cfg_.ny);

  const std::vector<int> key_iteration_values = analysis_key_iterations();
  const std::set<int> key_iterations(key_iteration_values.begin(), key_iteration_values.end());
  double previous_true_res_l2 = 0.0;
  int final_iteration = 0;
  double final_true_res_l2 = 0.0;
  double final_rel_true_res_l2 = 0.0;
  std::vector<double> final_solution;

  std::vector<double> x(static_cast<std::size_t>(matrix.n), 0.0);
  const LinearSolveReport report = ch_sparse_krylov::solve_preconditioned_cg(
      matrix, preconditioner, system.rhs, cfg_.poisson_iterations, cfg_.pressure_tolerance, x, project_vector,
      [&](const CGMonitorData &data) {
        const std::vector<double> &residual = *data.r;
        double true_res_linf = 0.0;
        double interface_sq = 0.0;
        double bulk_sq = 0.0;
        for (std::size_t idx = 0; idx < residual.size(); ++idx) {
          true_res_linf = std::max(true_res_linf, std::abs(residual[idx]));
          const double weighted = residual[idx] * residual[idx];
          if (interface_mask[idx] > 0.5) {
            interface_sq += weighted;
          } else {
            bulk_sq += weighted;
          }
        }
        const double interface_res_l2 = std::sqrt(interface_sq);
        const double bulk_res_l2 = std::sqrt(bulk_sq);
        const double q_k =
            (data.iteration == 0 || previous_true_res_l2 <= 0.0) ? 1.0 : data.absolute_residual / previous_true_res_l2;
        iter_out << cfg_.name << "," << cfg_.analysis_case_group << "," << snapshot.source_mode << ","
                 << snapshot.step << "," << snapshot.time << "," << cfg_.nx << "x" << cfg_.ny << ","
                 << cfg_.density_ratio << "," << compute_effective_sigma(cfg_) << "," << cfg_.viscosity_ratio << ","
                 << solver_name << "," << preconditioner_name(preconditioner.type) << "," << cfg_.poisson_iterations
                 << "," << cfg_.pressure_tolerance << "," << cfg_.analysis_initial_guess << "," << data.iteration
                 << "," << data.absolute_residual << "," << data.relative_residual << "," << true_res_linf << ","
                 << data.preconditioned_residual << "," << data.alpha << "," << data.beta << "," << q_k << ","
                 << interface_res_l2 << "," << bulk_res_l2 << ","
                 << (data.absolute_residual > 0.0 ? interface_res_l2 / data.absolute_residual : 0.0) << ","
                 << (data.absolute_residual > 0.0 ? bulk_res_l2 / data.absolute_residual : 0.0) << "\n";
        previous_true_res_l2 = data.absolute_residual;
        final_iteration = data.iteration;
        final_true_res_l2 = data.absolute_residual;
        final_rel_true_res_l2 = data.relative_residual;
        final_solution = *data.x;

        if (key_iterations.count(data.iteration) > 0) {
          std::ostringstream name;
          name << "residual_iter_" << std::setw(6) << std::setfill('0') << data.iteration << ".csv";
          write_matrix_csv(field_dir / name.str(), field_from_vector(residual, cfg_.nx, cfg_.ny), cfg_.nx, cfg_.ny);
        }
      });

  (void)report;
  if (final_solution.empty()) {
    final_solution = x;
  }
  std::vector<double> residual_final;
  ch_sparse_krylov::apply_matrix(matrix, final_solution, residual_final);
  for (std::size_t idx = 0; idx < residual_final.size(); ++idx) {
    residual_final[idx] = system.rhs[idx] - residual_final[idx];
  }
  if (project_vector) {
    project_vector(residual_final);
  }
  write_matrix_csv(field_dir / "residual_final.csv", field_from_vector(residual_final, cfg_.nx, cfg_.ny), cfg_.nx,
                   cfg_.ny);

  const double pressure_mean_before_fix = vector_mean(final_solution);
  if (project_vector) {
    project_vector(final_solution);
  }
  const double pressure_mean_after_fix = vector_mean(final_solution);

  std::ostringstream summary_row;
  summary_row << std::setprecision(17) << cfg_.name << "," << cfg_.analysis_case_group << ","
              << snapshot.source_mode << "," << snapshot.step << "," << snapshot.time << "," << cfg_.nx << "x"
              << cfg_.ny << "," << cfg_.density_ratio << "," << compute_effective_sigma(cfg_) << ","
              << cfg_.viscosity_ratio << "," << solver_name << "," << preconditioner_name(preconditioner.type) << ","
              << cfg_.poisson_iterations << "," << cfg_.pressure_tolerance << "," << cfg_.analysis_initial_guess << ","
              << vector_mean(system.rhs) << ","
              << std::accumulate(system.rhs.begin(), system.rhs.end(), 0.0) << "," << pressure_mean_before_fix << ","
              << pressure_mean_after_fix << "," << cfg_.analysis_nullspace_treatment << "," << final_iteration << ","
              << final_true_res_l2 << "," << final_rel_true_res_l2;
  append_summary_row(fs::path(pressure_analysis_dir()) / "summary.csv", summary_row.str());
}

void Solver::maybe_run_pressure_analysis_frozen() const {
  if (!analysis_mode_enabled("frozen")) {
    return;
  }
  if (current_step_index_ + 1 != cfg_.analysis_trigger_step) {
    return;
  }
  run_pressure_analysis_from_snapshot(make_pressure_analysis_snapshot("frozen"));
}

} // namespace ding
