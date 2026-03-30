#include "solver.hpp"
#include "internal.hpp"

#include <algorithm>
#include <cmath>
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
  Field2D c_mid = c_new;
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      c_mid(i, j) = 0.5 * (c_old(i, j) + c_new(i, j));
    }
  }
  apply_scalar_bc(c_mid);

  Field2D mu_mid(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0);
  update_chemical_potential(c_mid, mu_mid);

  const double scale = 1.0 / (cfg_.re * cfg_.ca);
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

void Solver::solve_phase_linear_system_eq25(const Field2D &rhs_field, double target_mean, Field2D &c_state,
                                            double &iterate_residual, double &equation_residual) const {
  const double alpha0 = 3.0 / (2.0 * cfg_.dt);
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

  double current_mean = 0.0;
  for (double value : x) {
    current_mean += value;
  }
  current_mean /= static_cast<double>(n);
  const double shift = target_mean - current_mean;

  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      const std::size_t row = static_cast<std::size_t>(row_index(i, j));
      c_state(i, j) = clamp_phase_debug(x[row] + shift);
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
  solve_phase_linear_system_eq25(rhs_field, target_mean, c_state, iterate_residual, equation_residual);

  c_ = c_state;
  apply_scalar_bc(c_);
  update_chemical_potential(c_, mu_);
  last_ch_equation_residual_ = equation_residual;
  phase_advection_rhs_prev_.fill(0.0);
  phase_explicit_operator_prev_ = explicit_operator_n;
  return iterate_residual;
}

} // namespace ding
