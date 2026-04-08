#include "solver.hpp"
#include "internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace ding {

using coupled_detail::square;

namespace {

const char *ch_preconditioner_name(ch_sparse_krylov::PreconditionerType type) {
  if (type == ch_sparse_krylov::PreconditionerType::incomplete_ldlt) {
    return "SparsePCG[ILDLT]";
  }
  if (type == ch_sparse_krylov::PreconditionerType::incomplete_cholesky) {
    return "SparsePCG[ICC]";
  }
  return "SparsePCG[Diagonal]";
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

void write_solver_vector_file(const std::filesystem::path &path, const std::vector<double> &values) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("cannot open solver vector file: " + path.string());
  }
  out << values.size() << "\n";
  out << std::setprecision(17);
  for (double value : values) {
    out << value << "\n";
  }
}

} // namespace

void Solver::update_materials_from_phase(const Field2D &c_state, Field2D &rho_state, Field2D &eta_state) const {
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      rho_state(i, j) = c_state(i, j) + (1.0 - c_state(i, j)) * cfg_.density_ratio;
      eta_state(i, j) = c_state(i, j) + (1.0 - c_state(i, j)) * cfg_.viscosity_ratio;
    }
  }
  apply_scalar_bc(rho_state);
  apply_scalar_bc(eta_state);
}

void Solver::update_materials() { update_materials_from_phase(c_, rho_, eta_); }

void Solver::update_midpoint_materials() {
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      rho_mid_(i, j) = 0.5 * (rho_previous_step_(i, j) + rho_(i, j));
      eta_mid_(i, j) = 0.5 * (eta_previous_step_(i, j) + eta_(i, j));
    }
  }
  apply_scalar_bc(rho_mid_);
  apply_scalar_bc(eta_mid_);
}

void Solver::update_chemical_potential(const Field2D &c_state, Field2D &mu_state) const {
  Field2D work = c_state;
  apply_scalar_bc(work);
  const double alpha = 6.0 * std::sqrt(2.0);
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      const double c = work(i, j);
      const double dw = 0.5 * c * (1.0 - c) * (1.0 - 2.0 * c);
      mu_state(i, j) = alpha * (dw / cfg_.cn - cfg_.cn * laplacian_center(work, i, j));
    }
  }
  apply_scalar_bc(mu_state);
}

void Solver::ensure_ch_operator_matrices() const {
  if (ch_operator_matrices_ready_) {
    return;
  }

  ch_laplacian_matrix_ =
      ch_sparse_krylov::build_laplacian_matrix(cfg_.nx, cfg_.ny, dx_, dy_, cfg_.periodic_x, cfg_.periodic_y);
  ch_biharmonic_matrix_ = ch_sparse_krylov::multiply(ch_laplacian_matrix_, ch_laplacian_matrix_);
  ch_operator_matrices_ready_ = true;
}

void Solver::ensure_ch_linear_system(double alpha0) const {
  ensure_ch_operator_matrices();
  if (std::abs(alpha0 - ch_cached_alpha0_) <= 1.0e-14 * std::max(1.0, std::abs(alpha0))) {
    return;
  }

  const double beta1 = cfg_.stabilization_a1 / cfg_.pe;
  const double beta2 = cfg_.stabilization_a2 / cfg_.pe;
  ch_linear_system_matrix_ =
      ch_sparse_krylov::build_cahn_hilliard_operator(ch_laplacian_matrix_, ch_biharmonic_matrix_, alpha0, beta1, beta2);
  ch_linear_system_preconditioner_ = ch_sparse_krylov::build_preconditioner(ch_linear_system_matrix_);
  ch_cached_alpha0_ = alpha0;
}

double Solver::weno5_left(double v1, double v2, double v3, double v4, double v5) const {
  const double eps = 1.0e-6;
  const double q0 = (2.0 * v1 - 7.0 * v2 + 11.0 * v3) / 6.0;
  const double q1 = (-v2 + 5.0 * v3 + 2.0 * v4) / 6.0;
  const double q2 = (2.0 * v3 + 5.0 * v4 - v5) / 6.0;
  const double b0 = 13.0 / 12.0 * square(v1 - 2.0 * v2 + v3) + 0.25 * square(v1 - 4.0 * v2 + 3.0 * v3);
  const double b1 = 13.0 / 12.0 * square(v2 - 2.0 * v3 + v4) + 0.25 * square(v2 - v4);
  const double b2 = 13.0 / 12.0 * square(v3 - 2.0 * v4 + v5) + 0.25 * square(3.0 * v3 - 4.0 * v4 + v5);
  const double a0 = 0.1 / square(eps + b0);
  const double a1 = 0.6 / square(eps + b1);
  const double a2 = 0.3 / square(eps + b2);
  const double sum = a0 + a1 + a2;
  return (a0 * q0 + a1 * q1 + a2 * q2) / sum;
}

double Solver::weno5_right(double v1, double v2, double v3, double v4, double v5) const {
  return weno5_left(v5, v4, v3, v2, v1);
}

double Solver::phase_weno_x_face_value(const Field2D &c_state, const Field2D &u_adv, int i, int j) const {
  // The same five-point WENO stencil is used at interior and non-periodic boundaries.
  // For non-periodic walls, scalar ghost cells are filled by apply_scalar_bc() with
  // zero-normal-gradient extension, so the face reconstruction remains a one-sided
  // WENO closure instead of falling back to first-order constant face values.
  return (u_adv(i, j) >= 0.0)
             ? weno5_left(c_state(i - 3, j), c_state(i - 2, j), c_state(i - 1, j), c_state(i, j), c_state(i + 1, j))
             : weno5_right(c_state(i - 2, j), c_state(i - 1, j), c_state(i, j), c_state(i + 1, j), c_state(i + 2, j));
}

double Solver::phase_weno_y_face_value(const Field2D &c_state, const Field2D &v_adv, int i, int j) const {
  return (v_adv(i, j) >= 0.0)
             ? weno5_left(c_state(i, j - 3), c_state(i, j - 2), c_state(i, j - 1), c_state(i, j), c_state(i, j + 1))
             : weno5_right(c_state(i, j - 2), c_state(i, j - 1), c_state(i, j), c_state(i, j + 1), c_state(i, j + 2));
}

void Solver::update_surface_tension_force(const Field2D &c_old, const Field2D &c_new) {
  if (cfg_.surface_tension_multiplier == 0.0) {
    surface_fx_cell_.fill(0.0);
    surface_fy_cell_.fill(0.0);
    surface_fx_u_.fill(0.0);
    surface_fy_v_.fill(0.0);
    apply_scalar_bc(surface_fx_cell_);
    apply_scalar_bc(surface_fy_cell_);
    apply_u_bc(surface_fx_u_);
    apply_v_bc(surface_fy_v_);
    return;
  }

  Field2D c_mid = c_new;
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      c_mid(i, j) = 0.5 * (c_old(i, j) + c_new(i, j));
    }
  }
  apply_scalar_bc(c_mid);

  Field2D mu_mid(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0);
  update_chemical_potential(c_mid, mu_mid);

  const double scale = cfg_.surface_tension_multiplier / (cfg_.re * cfg_.ca);
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      surface_fx_cell_(i, j) = scale * mu_mid(i, j) * grad_center_x(c_mid, i, j);
      surface_fy_cell_(i, j) = scale * mu_mid(i, j) * grad_center_y(c_mid, i, j);
    }
  }
  apply_scalar_bc(surface_fx_cell_);
  apply_scalar_bc(surface_fy_cell_);

  for (int i = 0; i < u_.nx; ++i) {
    for (int j = 0; j < u_.ny; ++j) {
      const double mu_face = (i <= 0)
                                 ? 0.5 * (mu_mid(cfg_.nx - 1, j) + mu_mid(0, j))
                                 : (i >= cfg_.nx ? 0.5 * (mu_mid(cfg_.nx - 1, j) + mu_mid(0, j))
                                                 : 0.5 * (mu_mid(i - 1, j) + mu_mid(i, j)));
      // Face force is built directly on the velocity control-volume face so that
      // force/rho_face is collocated with the pressure-gradient/rho_face term.
      surface_fx_u_(i, j) = scale * mu_face * grad_face_x_centered(c_mid, i, j);
    }
  }

  for (int i = 0; i < v_.nx; ++i) {
    for (int j = 0; j < v_.ny; ++j) {
      const double mu_face = (j <= 0)
                                 ? 0.5 * (mu_mid(i, cfg_.ny - 1) + mu_mid(i, 0))
                                 : (j >= cfg_.ny ? 0.5 * (mu_mid(i, cfg_.ny - 1) + mu_mid(i, 0))
                                                 : 0.5 * (mu_mid(i, j - 1) + mu_mid(i, j)));
      surface_fy_v_(i, j) = scale * mu_face * grad_face_y_centered(c_mid, i, j);
    }
  }
  apply_u_bc(surface_fx_u_);
  apply_v_bc(surface_fy_v_);

  if (cfg_.surface_tension_smoothing_passes <= 0 || cfg_.surface_tension_smoothing_weight <= 0.0) {
    return;
  }

  auto smooth_field = [&](Field2D &field, const auto &apply_bc) {
    Field2D scratch(field.nx, field.ny, field.ghost, 0.0);
    const double neighbor_weight = cfg_.surface_tension_smoothing_weight;
    const double center_weight = 1.0 - 4.0 * neighbor_weight;
    for (int pass = 0; pass < cfg_.surface_tension_smoothing_passes; ++pass) {
      apply_bc(field);
      for (int i = 0; i < field.nx; ++i) {
        for (int j = 0; j < field.ny; ++j) {
          scratch(i, j) = center_weight * field(i, j) +
                          neighbor_weight *
                              (field(i - 1, j) + field(i + 1, j) + field(i, j - 1) + field(i, j + 1));
        }
      }
      apply_bc(scratch);
      field = scratch;
    }
  };

  smooth_field(surface_fx_cell_, [&](Field2D &field) { apply_scalar_bc(field); });
  smooth_field(surface_fy_cell_, [&](Field2D &field) { apply_scalar_bc(field); });
  smooth_field(surface_fx_u_, [&](Field2D &field) { apply_u_bc(field); });
  smooth_field(surface_fy_v_, [&](Field2D &field) { apply_v_bc(field); });
}

void Solver::build_phase_advection_fluxes(const Field2D &c_state, const Field2D &u_adv, const Field2D &v_adv,
                                          Field2D &adv_flux_x, Field2D &adv_flux_y, Field2D &adv_rhs) const {
  Field2D c_ext = c_state;
  apply_scalar_bc(c_ext);

  for (int i = 0; i < cfg_.nx + 1; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      const double c_face = phase_weno_x_face_value(c_ext, u_adv, i, j);
      adv_flux_x(i, j) = u_adv(i, j) * c_face;
    }
  }

  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny + 1; ++j) {
      const double c_face = phase_weno_y_face_value(c_ext, v_adv, i, j);
      adv_flux_y(i, j) = v_adv(i, j) * c_face;
    }
  }
  apply_u_bc(adv_flux_x);
  apply_v_bc(adv_flux_y);

  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      adv_rhs(i, j) = -(adv_flux_x(i + 1, j) - adv_flux_x(i, j)) / dx_ -
                      (adv_flux_y(i, j + 1) - adv_flux_y(i, j)) / dy_;
    }
  }
  apply_scalar_bc(adv_rhs);
}

void Solver::build_phase_diffusion_fluxes(const Field2D &scalar_field, const Field2D &mobility, Field2D &flux_x,
                                          Field2D &flux_y, Field2D &divergence) const {
  for (int i = 0; i < cfg_.nx + 1; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      double m_face = 0.0;
      if (i <= 0) {
        m_face = cfg_.periodic_x ? 0.5 * (mobility(cfg_.nx - 1, j) + mobility(0, j)) : mobility(0, j);
      } else if (i >= cfg_.nx) {
        m_face = cfg_.periodic_x ? 0.5 * (mobility(cfg_.nx - 1, j) + mobility(0, j)) : mobility(cfg_.nx - 1, j);
      } else {
        m_face = 0.5 * (mobility(i - 1, j) + mobility(i, j));
      }
      flux_x(i, j) = m_face * grad_face_x_centered(scalar_field, i, j);
    }
  }

  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny + 1; ++j) {
      double m_face = 0.0;
      if (j <= 0) {
        m_face = cfg_.periodic_y ? 0.5 * (mobility(i, cfg_.ny - 1) + mobility(i, 0)) : mobility(i, 0);
      } else if (j >= cfg_.ny) {
        m_face = cfg_.periodic_y ? 0.5 * (mobility(i, cfg_.ny - 1) + mobility(i, 0)) : mobility(i, cfg_.ny - 1);
      } else {
        m_face = 0.5 * (mobility(i, j - 1) + mobility(i, j));
      }
      flux_y(i, j) = m_face * grad_face_y_centered(scalar_field, i, j);
    }
  }
  apply_u_bc(flux_x);
  apply_v_bc(flux_y);

  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      divergence(i, j) = (flux_x(i + 1, j) - flux_x(i, j)) / dx_ +
                         (flux_y(i, j + 1) - flux_y(i, j)) / dy_;
    }
  }
  apply_scalar_bc(divergence);
}

void Solver::build_phase_explicit_operator(const Field2D &c_state, const Field2D &u_adv, const Field2D &v_adv,
                                           Field2D &explicit_operator) const {
  Field2D mobility(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0);
  Field2D mu_state(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0);
  Field2D adv_flux_x(cfg_.nx + 1, cfg_.ny, cfg_.ghost, 0.0);
  Field2D adv_flux_y(cfg_.nx, cfg_.ny + 1, cfg_.ghost, 0.0);
  Field2D adv_rhs(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0);
  Field2D diff_flux_x(cfg_.nx + 1, cfg_.ny, cfg_.ghost, 0.0);
  Field2D diff_flux_y(cfg_.nx, cfg_.ny + 1, cfg_.ghost, 0.0);
  Field2D diff_div(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0);
  Field2D lap_c(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0);
  Field2D biharm_c(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0);

  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      mobility(i, j) = std::max(0.0, c_state(i, j) * (1.0 - c_state(i, j)));
      lap_c(i, j) = laplacian_center(c_state, i, j);
    }
  }
  apply_scalar_bc(mobility);
  apply_scalar_bc(lap_c);
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      biharm_c(i, j) = laplacian_center(lap_c, i, j);
    }
  }
  apply_scalar_bc(biharm_c);

  update_chemical_potential(c_state, mu_state);
  build_phase_advection_fluxes(c_state, u_adv, v_adv, adv_flux_x, adv_flux_y, adv_rhs);
  build_phase_diffusion_fluxes(mu_state, mobility, diff_flux_x, diff_flux_y, diff_div);

  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      explicit_operator(i, j) =
          diff_div(i, j) / cfg_.pe -
          (cfg_.stabilization_a1 * lap_c(i, j) - cfg_.stabilization_a2 * biharm_c(i, j)) / cfg_.pe + adv_rhs(i, j);
    }
  }
  apply_scalar_bc(explicit_operator);
}

double Solver::solve_phase_advection_only(const Field2D &u_adv, const Field2D &v_adv) {
  Field2D adv_flux_x(cfg_.nx + 1, cfg_.ny, cfg_.ghost, 0.0);
  Field2D adv_flux_y(cfg_.nx, cfg_.ny + 1, cfg_.ghost, 0.0);
  Field2D rhs_stage(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0);
  Field2D c0 = c_;
  Field2D c1 = c_;
  Field2D c2 = c_;

  build_phase_advection_fluxes(c0, u_adv, v_adv, adv_flux_x, adv_flux_y, rhs_stage);
  phase_advection_rhs_ = rhs_stage;
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      c1(i, j) = c0(i, j) + cfg_.dt * rhs_stage(i, j);
    }
  }
  apply_scalar_bc(c1);

  build_phase_advection_fluxes(c1, u_adv, v_adv, adv_flux_x, adv_flux_y, rhs_stage);
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      const double predictor = c1(i, j) + cfg_.dt * rhs_stage(i, j);
      c2(i, j) = 0.75 * c0(i, j) + 0.25 * predictor;
    }
  }
  apply_scalar_bc(c2);

  build_phase_advection_fluxes(c2, u_adv, v_adv, adv_flux_x, adv_flux_y, rhs_stage);
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      const double predictor = c2(i, j) + cfg_.dt * rhs_stage(i, j);
      c_(i, j) = (1.0 / 3.0) * c0(i, j) + (2.0 / 3.0) * predictor;
    }
  }
  apply_scalar_bc(c_);
  phase_advection_rhs_prev_ = rhs_stage;

  return field_diff_l2(c_, c0, 0, cfg_.nx, 0, cfg_.ny);
}

void Solver::solve_phase_linear_system_eq25(const Field2D &rhs_field, double alpha0, double target_mean,
                                            Field2D &c_state, double &iterate_residual,
                                            double &equation_residual) const {
  (void)target_mean;
  if (cfg_.ch_solver == "petsc_pcg") {
    solve_phase_linear_system_eq25_petsc(rhs_field, alpha0, target_mean, c_state, iterate_residual, equation_residual);
    return;
  }

  ensure_ch_linear_system(alpha0);

  const int n = cfg_.nx * cfg_.ny;
  auto row_index = [this](int i, int j) { return i * cfg_.ny + j; };

  std::vector<double> rhs(static_cast<std::size_t>(n), 0.0);
  std::vector<double> x(static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      const std::size_t row = static_cast<std::size_t>(row_index(i, j));
      rhs[row] = rhs_field(i, j);
      x[row] = c_state(i, j);
    }
  }

  iterate_residual = 0.0;
  equation_residual = 0.0;
  ch_sparse_krylov::LinearSolveReport report;
  last_ch_solver_name_ = ch_preconditioner_name(ch_linear_system_preconditioner_.type);
  try {
    report = ch_sparse_krylov::solve_preconditioned_cg(ch_linear_system_matrix_, ch_linear_system_preconditioner_, rhs,
                                                       cfg_.ch_inner_iterations, cfg_.ch_tolerance, x);
  } catch (const std::runtime_error &) {
    ch_linear_system_preconditioner_ = ch_sparse_krylov::build_diagonal_preconditioner(ch_linear_system_matrix_);
    last_ch_solver_name_ = "SparsePCG[DiagonalFallback]";
    report = ch_sparse_krylov::solve_preconditioned_cg(ch_linear_system_matrix_, ch_linear_system_preconditioner_, rhs,
                                                       cfg_.ch_inner_iterations, cfg_.ch_tolerance, x);
  }
  last_ch_iterations_ = report.iterations;

  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      const std::size_t row = static_cast<std::size_t>(row_index(i, j));
      c_state(i, j) = x[row];
    }
  }
  apply_scalar_bc(c_state);

  std::vector<double> corrected_x(static_cast<std::size_t>(n), 0.0);
  double rhs_norm_sq = 0.0;
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      const std::size_t row = static_cast<std::size_t>(row_index(i, j));
      corrected_x[row] = c_state(i, j);
      rhs_norm_sq += square(rhs[row]);
    }
  }

  std::vector<double> ax;
  ch_sparse_krylov::apply_matrix(ch_linear_system_matrix_, corrected_x, ax);
  double residual_sq = 0.0;
  for (int idx = 0; idx < n; ++idx) {
    residual_sq += square(rhs[static_cast<std::size_t>(idx)] - ax[static_cast<std::size_t>(idx)]);
  }

  iterate_residual = report.iterate_residual;
  equation_residual = std::sqrt(residual_sq / std::max(rhs_norm_sq, 1.0e-30));
}

void Solver::solve_phase_linear_system_eq25_petsc(const Field2D &rhs_field, double alpha0, double target_mean,
                                                  Field2D &c_state, double &iterate_residual,
                                                  double &equation_residual) const {
  (void)target_mean;
  namespace fs = std::filesystem;

  ensure_ch_linear_system(alpha0);

  const int n = cfg_.nx * cfg_.ny;
  auto row_index = [this](int i, int j) { return i * cfg_.ny + j; };

  std::vector<double> rhs(static_cast<std::size_t>(n), 0.0);
  std::vector<double> x(static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      const std::size_t row = static_cast<std::size_t>(row_index(i, j));
      rhs[row] = rhs_field(i, j);
      x[row] = c_state(i, j);
    }
  }

  const fs::path solver_dir = ch_solver_dir();
  fs::create_directories(solver_dir);
  const fs::path matrix_path = solver_dir / "matrix_triplets.txt";
  const fs::path rhs_path = solver_dir / "rhs.txt";
  const fs::path guess_path = solver_dir / "initial_guess.txt";
  const fs::path solution_path = solver_dir / "solution.txt";
  const fs::path report_path = solver_dir / "report.txt";
  const bool write_monitor_log =
      cfg_.petsc_ch_log_every > 0 && ((current_step_index_ + 1) % cfg_.petsc_ch_log_every == 0);
  std::ostringstream monitor_name;
  monitor_name << "residual_step_" << std::setw(6) << std::setfill('0') << (current_step_index_ + 1) << ".log";
  const fs::path monitor_log_path = solver_dir / monitor_name.str();

  {
    std::ofstream matrix_out(matrix_path);
    if (!matrix_out) {
      throw std::runtime_error("cannot open PETSc CH matrix file: " + matrix_path.string());
    }
    matrix_out << n << " " << n << " " << ch_linear_system_matrix_.values.size() << "\n";
    matrix_out << std::setprecision(17);
    for (int row = 0; row < ch_linear_system_matrix_.n; ++row) {
      for (int pos = ch_linear_system_matrix_.row_ptr[static_cast<std::size_t>(row)];
           pos < ch_linear_system_matrix_.row_ptr[static_cast<std::size_t>(row + 1)]; ++pos) {
        matrix_out << row << " " << ch_linear_system_matrix_.col_idx[static_cast<std::size_t>(pos)] << " "
                   << ch_linear_system_matrix_.values[static_cast<std::size_t>(pos)] << "\n";
      }
    }
  }

  write_solver_vector_file(rhs_path, rhs);
  write_solver_vector_file(guess_path, x);

  const fs::path script_path = cfg_.petsc_solver_script;
  const fs::path options_path = cfg_.petsc_solver_config;
  if (!fs::exists(script_path)) {
    throw std::runtime_error("PETSc CH helper script not found: " + script_path.string());
  }
  if (!fs::exists(options_path)) {
    throw std::runtime_error("PETSc CH options file not found: " + options_path.string());
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
          << shell_quote(matrix_path.string()) << " --rhs " << shell_quote(rhs_path.string()) << " --initial-guess "
          << shell_quote(guess_path.string()) << " --solution " << shell_quote(solution_path.string()) << " --report "
          << shell_quote(report_path.string()) << " --config " << shell_quote(options_path.string())
          << " --use-constant-nullspace false";
  if (cfg_.verbose) {
    command << " --log-prefix \"[ch step " << (current_step_index_ + 1) << "]\"";
  }
  if (write_monitor_log) {
    command << " --monitor-log " << shell_quote(monitor_log_path.string());
  }

  const int code = std::system(command.str().c_str());
  if (code != 0) {
    throw std::runtime_error("PETSc CH solve failed with exit code " + std::to_string(code));
  }

  {
    std::ifstream solution_in(solution_path);
    if (!solution_in) {
      throw std::runtime_error("cannot open PETSc CH solution file: " + solution_path.string());
    }
    int read_n = 0;
    solution_in >> read_n;
    if (read_n != n) {
      throw std::runtime_error("PETSc CH solution size mismatch");
    }
    for (int idx = 0; idx < n; ++idx) {
      solution_in >> x[static_cast<std::size_t>(idx)];
    }
  }

  iterate_residual = 0.0;
  equation_residual = 0.0;
  last_ch_solver_name_ = "PETSc[CG+ICC]";
  {
    std::ifstream report_in(report_path);
    if (!report_in) {
      throw std::runtime_error("cannot open PETSc CH report file: " + report_path.string());
    }
    std::string key;
    std::string ksp_type;
    std::string pc_type;
    while (report_in >> key) {
      if (key == "residual_norm") {
        report_in >> iterate_residual;
      } else if (key == "iterations") {
        report_in >> last_ch_iterations_;
      } else if (key == "ksp_type") {
        report_in >> ksp_type;
      } else if (key == "pc_type") {
        report_in >> pc_type;
      } else {
        std::string value;
        report_in >> value;
      }
    }
    if (!ksp_type.empty() || !pc_type.empty()) {
      last_ch_solver_name_ = "PETSc[" + ksp_type + "+" + pc_type + "]";
    }
  }

  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      const std::size_t row = static_cast<std::size_t>(row_index(i, j));
      c_state(i, j) = x[row];
    }
  }
  apply_scalar_bc(c_state);

  std::vector<double> corrected_x(static_cast<std::size_t>(n), 0.0);
  double rhs_norm_sq = 0.0;
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      const std::size_t row = static_cast<std::size_t>(row_index(i, j));
      corrected_x[row] = c_state(i, j);
      rhs_norm_sq += square(rhs[row]);
    }
  }

  std::vector<double> ax;
  ch_sparse_krylov::apply_matrix(ch_linear_system_matrix_, corrected_x, ax);
  double residual_sq = 0.0;
  for (int idx = 0; idx < n; ++idx) {
    residual_sq += square(rhs[static_cast<std::size_t>(idx)] - ax[static_cast<std::size_t>(idx)]);
  }
  equation_residual = std::sqrt(residual_sq / std::max(rhs_norm_sq, 1.0e-30));
}

double Solver::solve_cahn_hilliard_semi_implicit(const Field2D &u_adv, const Field2D &v_adv, int step) {
  Field2D rhs_field(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0);
  Field2D explicit_operator_n(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0);
  Field2D c_state = c_;

  // Ding 2007 Eq. (25):
  // (3 C^{n+1} - 4 C^n + C^{n-1}) / (2 dt)
  //   = (a1 Lap C^{n+1} - a2 Lap^2 C^{n+1}) / Pe + [ 2 A(C^n,u^n) - A(C^{n-1},u^{n-1}) ]
  // with A defined by Eq. (26).
  // This implementation matches that split BDF2/AB2/stabilized structure.
  // The implicit CH subproblem is assembled as a sparse matrix and solved with a Krylov method.
  build_phase_explicit_operator(c_, u_adv, v_adv, explicit_operator_n);
  const double alpha0 = (step == 0) ? (1.0 / cfg_.dt) : (3.0 / (2.0 * cfg_.dt));

  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      if (step == 0) {
        rhs_field(i, j) = c_previous_step_(i, j) / cfg_.dt + explicit_operator_n(i, j);
      } else {
        rhs_field(i, j) = (4.0 * c_previous_step_(i, j) - c_two_steps_back_(i, j)) / (2.0 * cfg_.dt) +
                          2.0 * explicit_operator_n(i, j) - phase_explicit_operator_prev_(i, j);
      }
    }
  }
  apply_scalar_bc(rhs_field);

  const double target_mean = (step == 0)
                                 ? compute_mass(c_previous_step_) / (cfg_.lx * cfg_.ly)
                                 : (4.0 * compute_mass(c_previous_step_) - compute_mass(c_two_steps_back_)) /
                                       (3.0 * cfg_.lx * cfg_.ly);
  double iterate_residual = 0.0;
  double equation_residual = 0.0;
  solve_phase_linear_system_eq25(rhs_field, alpha0, target_mean, c_state, iterate_residual, equation_residual);

  c_ = c_state;
  apply_scalar_bc(c_);
  update_chemical_potential(c_, mu_);
  last_ch_equation_residual_ = equation_residual;
  phase_advection_rhs_prev_.fill(0.0);
  phase_explicit_operator_prev_ = explicit_operator_n;
  return iterate_residual;
}

} // namespace ding
