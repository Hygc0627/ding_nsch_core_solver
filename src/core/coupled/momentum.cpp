#include "solver.hpp"
#include "internal.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace ding {

using coupled_detail::square;

namespace {

const char *momentum_preconditioner_name(ch_sparse_krylov::PreconditionerType type) {
  if (type == ch_sparse_krylov::PreconditionerType::incomplete_cholesky) {
    return "ICC";
  }
  if (type == ch_sparse_krylov::PreconditionerType::incomplete_ldlt) {
    return "ILDLT";
  }
  return "Diagonal";
}

} // namespace

double Solver::second_order_upwind(double vel, double q_mm, double q_m, double q_p, double q_pp) const {
  if (vel >= 0.0) {
    return 0.5 * (3.0 * q_m - q_mm);
  }
  return 0.5 * (3.0 * q_p - q_pp);
}


double Solver::reconstruct_with_scheme(double vel, double q_mm, double q_m, double q_p, double q_pp) const {
  if (cfg_.momentum_advection_scheme == "centered") {
    return 0.5 * (q_m + q_p);
  }
  if (cfg_.momentum_advection_scheme == "donor-cell") {
    return vel >= 0.0 ? q_m : q_p;
  }
  return second_order_upwind(vel, q_mm, q_m, q_p, q_pp);
}

double Solver::momentum_u_face_value(const Field2D &u_state, int i, int j) const {
  return reconstruct_with_scheme(u_state(i, j), u_state(i - 2, j), u_state(i - 1, j), u_state(i, j), u_state(i + 1, j));
}

double Solver::momentum_v_face_value(const Field2D &v_state, int i, int j) const {
  return reconstruct_with_scheme(v_state(i, j), v_state(i, j - 2), v_state(i, j - 1), v_state(i, j), v_state(i, j + 1));
}


void Solver::compute_momentum_fluxes(Field2D &u_adv, Field2D &v_adv) const {
  u_adv.fill(0.0);
  v_adv.fill(0.0);

  for (int i = 0; i < u_.nx; ++i) {
    for (int j = 0; j < u_.ny; ++j) {
      // Ding 2007 §3.2: momentum convection is discretized with central finite differences
      // on the staggered grid. Here H(u) approximates u du/dx + v du/dy on the u-face.
      const double u_face = u_(i, j);
      const double du_dx = (u_(i + 1, j) - u_(i - 1, j)) / (2.0 * dx_);
      const double du_dy = (u_(i, j + 1) - u_(i, j - 1)) / (2.0 * dy_);
      const double v_face =
          0.25 * (v_(i - 1, j) + v_(i, j) + v_(i - 1, j + 1) + v_(i, j + 1));
      u_adv(i, j) = u_face * du_dx + v_face * du_dy;
    }
  }

  for (int i = 0; i < v_.nx; ++i) {
    for (int j = 0; j < v_.ny; ++j) {
      // H(v) approximates u dv/dx + v dv/dy on the v-face.
      const double v_face = v_(i, j);
      const double dv_dx = (v_(i + 1, j) - v_(i - 1, j)) / (2.0 * dx_);
      const double dv_dy = (v_(i, j + 1) - v_(i, j - 1)) / (2.0 * dy_);
      const double u_face =
          0.25 * (u_(i, j - 1) + u_(i, j) + u_(i + 1, j - 1) + u_(i + 1, j));
      v_adv(i, j) = u_face * dv_dx + v_face * dv_dy;
    }
  }
  apply_u_bc(u_adv);
  apply_v_bc(v_adv);
}

double Solver::stress_divergence_u(const Field2D &u_state, const Field2D &v_state, int i, int j) const {
  return stress_divergence_u(u_state, v_state, eta_, i, j);
}

double Solver::viscous_self_u(const Field2D &u_state, const Field2D &eta_field, int i, int j) const {
  const double eta_e = eta_u_face(eta_field, i + 1, j);
  const double eta_w = eta_u_face(eta_field, i, j);
  const double dudx_e = (u_state(i + 1, j) - u_state(i, j)) / dx_;
  const double dudx_w = (u_state(i, j) - u_state(i - 1, j)) / dx_;
  const double normal = (2.0 * eta_e * dudx_e - 2.0 * eta_w * dudx_w) / dx_;

  const double eta_n = eta_corner(eta_field, i, j + 1);
  const double eta_s = eta_corner(eta_field, i, j);
  const double dudy_n = (u_state(i, j + 1) - u_state(i, j)) / dy_;
  const double dudy_s = (u_state(i, j) - u_state(i, j - 1)) / dy_;
  const double shear_self = (eta_n * dudy_n - eta_s * dudy_s) / dy_;

  return normal + shear_self;
}

double Solver::viscous_cross_u(const Field2D &v_state, const Field2D &eta_field, int i, int j) const {
  const double eta_n = eta_corner(eta_field, i, j + 1);
  const double eta_s = eta_corner(eta_field, i, j);
  const double dvdx_n = (v_state(i, j + 1) - v_state(i - 1, j + 1)) / dx_;
  const double dvdx_s = (v_state(i, j) - v_state(i - 1, j)) / dx_;
  return (eta_n * dvdx_n - eta_s * dvdx_s) / dy_;
}

double Solver::stress_divergence_u(const Field2D &u_state, const Field2D &v_state, const Field2D &eta_field, int i,
                                   int j) const {
  // u-control volume:
  // d/dx [ 2 eta du/dx ] + d/dy [ eta (du/dy + dv/dx) ]
  return viscous_self_u(u_state, eta_field, i, j) + viscous_cross_u(v_state, eta_field, i, j);
}

double Solver::stress_divergence_v(const Field2D &u_state, const Field2D &v_state, int i, int j) const {
  return stress_divergence_v(u_state, v_state, eta_, i, j);
}

double Solver::viscous_self_v(const Field2D &v_state, const Field2D &eta_field, int i, int j) const {
  const double eta_e = eta_corner(eta_field, i + 1, j);
  const double eta_w = eta_corner(eta_field, i, j);
  const double dvdx_e = (v_state(i + 1, j) - v_state(i, j)) / dx_;
  const double dvdx_w = (v_state(i, j) - v_state(i - 1, j)) / dx_;
  const double shear_self = (eta_e * dvdx_e - eta_w * dvdx_w) / dx_;

  const double eta_n = eta_v_face(eta_field, i, j + 1);
  const double eta_s = eta_v_face(eta_field, i, j);
  const double dvdy_n = (v_state(i, j + 1) - v_state(i, j)) / dy_;
  const double dvdy_s = (v_state(i, j) - v_state(i, j - 1)) / dy_;
  const double normal = (2.0 * eta_n * dvdy_n - 2.0 * eta_s * dvdy_s) / dy_;

  return shear_self + normal;
}

double Solver::viscous_cross_v(const Field2D &u_state, const Field2D &eta_field, int i, int j) const {
  const double eta_e = eta_corner(eta_field, i + 1, j);
  const double eta_w = eta_corner(eta_field, i, j);
  const double dudy_e = (u_state(i + 1, j) - u_state(i + 1, j - 1)) / dy_;
  const double dudy_w = (u_state(i, j) - u_state(i, j - 1)) / dy_;
  return (eta_e * dudy_e - eta_w * dudy_w) / dx_;
}

double Solver::stress_divergence_v(const Field2D &u_state, const Field2D &v_state, const Field2D &eta_field, int i,
                                   int j) const {
  // v-control volume:
  // d/dx [ eta (dv/dx + du/dy) ] + d/dy [ 2 eta dv/dy ]
  return viscous_self_v(v_state, eta_field, i, j) + viscous_cross_v(u_state, eta_field, i, j);
}

double Solver::solve_momentum_predictor(const Field2D &u_adv, const Field2D &v_adv, int step) {
  using ch_sparse_krylov::KrylovPreconditioner;
  using ch_sparse_krylov::LinearSolveReport;
  using ch_sparse_krylov::SparseMatrixCSR;

  const double alpha = 0.5 * cfg_.dt / cfg_.re;

  Field2D u_rhs(u_.nx, u_.ny, u_.ghost, 0.0);
  Field2D v_rhs(v_.nx, v_.ny, v_.ghost, 0.0);

  for (int i = 0; i < u_.nx; ++i) {
    for (int j = 0; j < u_.ny; ++j) {
      const double rho_face = rho_u_face(rho_mid_, i, j);
      const double conv = (step == 0) ? u_adv(i, j) : 1.5 * u_adv(i, j) - 0.5 * momentum_u_rhs_prev_(i, j);
      const double visc_old = stress_divergence_u(u_, v_, eta_previous_step_, i, j);
      u_rhs(i, j) = rho_face * u_(i, j) +
                    cfg_.dt * (-rho_face * conv + rho_face * cfg_.body_force_x + surface_fx_u_(i, j)) +
                    alpha * visc_old;
    }
  }

  for (int i = 0; i < v_.nx; ++i) {
    for (int j = 0; j < v_.ny; ++j) {
      const double rho_face = rho_v_face(rho_mid_, i, j);
      const double conv = (step == 0) ? v_adv(i, j) : 1.5 * v_adv(i, j) - 0.5 * momentum_v_rhs_prev_(i, j);
      const double visc_old = stress_divergence_v(u_, v_, eta_previous_step_, i, j);
      v_rhs(i, j) = rho_face * v_(i, j) +
                    cfg_.dt * (-rho_face * conv + rho_face * cfg_.body_force_y + surface_fy_v_(i, j)) +
                    alpha * visc_old;
    }
  }

  auto u_is_active = [&](int i, int j) {
    if (j < 0 || j >= u_.ny) {
      return false;
    }
    if (cfg_.periodic_x) {
      if (i < 0 || i >= cfg_.nx) {
        return false;
      }
    } else if (i <= 0 || i >= cfg_.nx) {
      return false;
    }
    return true;
  };

  auto v_is_active = [&](int i, int j) {
    if (i < 0 || i >= v_.nx) {
      return false;
    }
    if (cfg_.periodic_y) {
      if (j < 0 || j >= cfg_.ny) {
        return false;
      }
    } else if (j <= 0 || j >= cfg_.ny) {
      return false;
    }
    return true;
  };

  auto canonical_u_i = [&](int i) {
    if (!cfg_.periodic_x) {
      return i;
    }
    if (i < 0) {
      return cfg_.nx - 1;
    }
    if (i >= cfg_.nx) {
      return 0;
    }
    return i;
  };

  auto canonical_u_j = [&](int j) {
    if (!cfg_.periodic_y) {
      return j;
    }
    if (j < 0) {
      return cfg_.ny - 1;
    }
    if (j >= cfg_.ny) {
      return 0;
    }
    return j;
  };

  auto canonical_v_i = [&](int i) {
    if (!cfg_.periodic_x) {
      return i;
    }
    if (i < 0) {
      return v_.nx - 1;
    }
    if (i >= v_.nx) {
      return 0;
    }
    return i;
  };

  auto canonical_v_j = [&](int j) {
    if (!cfg_.periodic_y) {
      return j;
    }
    if (j < 0) {
      return cfg_.ny - 1;
    }
    if (j >= cfg_.ny) {
      return 0;
    }
    return j;
  };

  auto u_boundary_value = [&](int i, int j) {
    if (!cfg_.periodic_x && (i <= 0 || i >= cfg_.nx)) {
      return 0.0;
    }
    if (!cfg_.periodic_y) {
      if (j <= 0) {
        return cfg_.bottom_wall_velocity_x;
      }
      if (j >= cfg_.ny - 1) {
        return cfg_.top_wall_velocity_x;
      }
    }
    return 0.0;
  };

  auto v_boundary_value = [&](int, int) { return 0.0; };

  std::vector<int> u_row_of(static_cast<std::size_t>(u_.nx * u_.ny), -1);
  std::vector<int> v_row_of(static_cast<std::size_t>(v_.nx * v_.ny), -1);
  auto u_flat = [&](int i, int j) { return i * u_.ny + j; };
  auto v_flat = [&](int i, int j) { return i * v_.ny + j; };

  int u_unknowns = 0;
  for (int i = 0; i < u_.nx; ++i) {
    for (int j = 0; j < u_.ny; ++j) {
      if (u_is_active(i, j)) {
        u_row_of[static_cast<std::size_t>(u_flat(i, j))] = u_unknowns++;
      }
    }
  }

  int v_unknowns = 0;
  for (int i = 0; i < v_.nx; ++i) {
    for (int j = 0; j < v_.ny; ++j) {
      if (v_is_active(i, j)) {
        v_row_of[static_cast<std::size_t>(v_flat(i, j))] = v_unknowns++;
      }
    }
  }

  const int total_unknowns = u_unknowns + v_unknowns;
  std::vector<std::map<int, double>> row_maps(static_cast<std::size_t>(total_unknowns));
  std::vector<double> rhs(static_cast<std::size_t>(total_unknowns), 0.0);
  std::vector<double> x(static_cast<std::size_t>(total_unknowns), 0.0);

  auto u_row = [&](int i, int j) { return u_row_of[static_cast<std::size_t>(u_flat(i, j))]; };
  auto v_row = [&](int i, int j) { return u_unknowns + v_row_of[static_cast<std::size_t>(v_flat(i, j))]; };

  auto add_u_entry = [&](int row, int ui, int uj, double value) {
    if (u_is_active(ui, uj)) {
      row_maps[static_cast<std::size_t>(row)][u_row(ui, uj)] += value;
    } else {
      rhs[static_cast<std::size_t>(row)] -= value * u_boundary_value(ui, uj);
    }
  };

  auto add_u_y_entry = [&](int row, int ui, int uj, double value, int owner_i, int owner_j) {
    if (u_is_active(ui, uj)) {
      row_maps[static_cast<std::size_t>(row)][u_row(ui, uj)] += value;
      return;
    }
    if (!cfg_.periodic_y && (uj < 0 || uj >= u_.ny)) {
      const double wall_speed = (uj < 0) ? cfg_.bottom_wall_velocity_x : cfg_.top_wall_velocity_x;
      row_maps[static_cast<std::size_t>(row)][u_row(owner_i, owner_j)] -= value;
      rhs[static_cast<std::size_t>(row)] -= value * (2.0 * wall_speed);
      return;
    }
    rhs[static_cast<std::size_t>(row)] -= value * u_boundary_value(ui, uj);
  };

  auto add_v_entry = [&](int row, int vi, int vj, double value) {
    if (v_is_active(vi, vj)) {
      row_maps[static_cast<std::size_t>(row)][v_row(vi, vj)] += value;
    } else {
      rhs[static_cast<std::size_t>(row)] -= value * v_boundary_value(vi, vj);
    }
  };

  auto add_v_x_entry = [&](int row, int vi, int vj, double value, int owner_i, int owner_j) {
    if (v_is_active(vi, vj)) {
      row_maps[static_cast<std::size_t>(row)][v_row(vi, vj)] += value;
      return;
    }
    if (!cfg_.periodic_x && (vi < 0 || vi >= v_.nx)) {
      row_maps[static_cast<std::size_t>(row)][v_row(owner_i, owner_j)] -= value;
      return;
    }
    rhs[static_cast<std::size_t>(row)] -= value * v_boundary_value(vi, vj);
  };

  for (int i = 0; i < u_.nx; ++i) {
    for (int j = 0; j < u_.ny; ++j) {
      if (!u_is_active(i, j)) {
        continue;
      }

      const int row = u_row(i, j);
      rhs[static_cast<std::size_t>(row)] = u_rhs(i, j);
      x[static_cast<std::size_t>(row)] = u_(i, j);

      const double rho_face = rho_u_face(rho_mid_, i, j);
      const double eta_e = eta_u_face(eta_, i + 1, j);
      const double eta_w = eta_u_face(eta_, i, j);
      const double eta_n = eta_corner(eta_, i, j + 1);
      const double eta_s = eta_corner(eta_, i, j);
      const double coeff_e = 2.0 * eta_e / (dx_ * dx_);
      const double coeff_w = 2.0 * eta_w / (dx_ * dx_);
      const double coeff_n = eta_n / (dy_ * dy_);
      const double coeff_s = eta_s / (dy_ * dy_);
      const double cross_n = eta_n / (dx_ * dy_);
      const double cross_s = eta_s / (dx_ * dy_);

      row_maps[static_cast<std::size_t>(row)][row] += rho_face + alpha * (coeff_e + coeff_w + coeff_n + coeff_s);

      add_u_entry(row, canonical_u_i(i + 1), j, -alpha * coeff_e);
      add_u_entry(row, canonical_u_i(i - 1), j, -alpha * coeff_w);
      add_u_y_entry(row, i, canonical_u_j(j + 1), -alpha * coeff_n, i, j);
      add_u_y_entry(row, i, canonical_u_j(j - 1), -alpha * coeff_s, i, j);

      add_v_entry(row, canonical_v_i(i), canonical_v_j(j + 1), -alpha * cross_n);
      add_v_entry(row, canonical_v_i(i - 1), canonical_v_j(j + 1), alpha * cross_n);
      add_v_entry(row, canonical_v_i(i), canonical_v_j(j), alpha * cross_s);
      add_v_entry(row, canonical_v_i(i - 1), canonical_v_j(j), -alpha * cross_s);
    }
  }

  for (int i = 0; i < v_.nx; ++i) {
    for (int j = 0; j < v_.ny; ++j) {
      if (!v_is_active(i, j)) {
        continue;
      }

      const int row = v_row(i, j);
      rhs[static_cast<std::size_t>(row)] = v_rhs(i, j);
      x[static_cast<std::size_t>(row)] = v_(i, j);

      const double rho_face = rho_v_face(rho_mid_, i, j);
      const double eta_e = eta_corner(eta_, i + 1, j);
      const double eta_w = eta_corner(eta_, i, j);
      const double eta_n = eta_v_face(eta_, i, j + 1);
      const double eta_s = eta_v_face(eta_, i, j);
      const double coeff_e = eta_e / (dx_ * dx_);
      const double coeff_w = eta_w / (dx_ * dx_);
      const double coeff_n = 2.0 * eta_n / (dy_ * dy_);
      const double coeff_s = 2.0 * eta_s / (dy_ * dy_);
      const double cross_e = eta_e / (dx_ * dy_);
      const double cross_w = eta_w / (dx_ * dy_);

      row_maps[static_cast<std::size_t>(row)][row] += rho_face + alpha * (coeff_e + coeff_w + coeff_n + coeff_s);

      add_v_x_entry(row, canonical_v_i(i + 1), j, -alpha * coeff_e, i, j);
      add_v_x_entry(row, canonical_v_i(i - 1), j, -alpha * coeff_w, i, j);
      add_v_entry(row, i, canonical_v_j(j + 1), -alpha * coeff_n);
      add_v_entry(row, i, canonical_v_j(j - 1), -alpha * coeff_s);

      add_u_entry(row, canonical_u_i(i + 1), canonical_u_j(j), -alpha * cross_e);
      add_u_entry(row, canonical_u_i(i + 1), canonical_u_j(j - 1), alpha * cross_e);
      add_u_entry(row, canonical_u_i(i), canonical_u_j(j), alpha * cross_w);
      add_u_entry(row, canonical_u_i(i), canonical_u_j(j - 1), -alpha * cross_w);
    }
  }

  SparseMatrixCSR matrix;
  ch_sparse_krylov::finalize_row_maps(row_maps, matrix);
  const KrylovPreconditioner preconditioner = ch_sparse_krylov::build_diagonal_preconditioner(matrix);
  const LinearSolveReport report =
      ch_sparse_krylov::solve_preconditioned_bicgstab(matrix, preconditioner, rhs, cfg_.momentum_iterations,
                                                      cfg_.momentum_tolerance, x, false);

  u_star_ = u_;
  v_star_ = v_;
  for (int i = 0; i < u_.nx; ++i) {
    for (int j = 0; j < u_.ny; ++j) {
      if (u_is_active(i, j)) {
        u_star_(i, j) = x[static_cast<std::size_t>(u_row(i, j))];
      }
    }
  }
  for (int i = 0; i < v_.nx; ++i) {
    for (int j = 0; j < v_.ny; ++j) {
      if (v_is_active(i, j)) {
        v_star_(i, j) = x[static_cast<std::size_t>(v_row(i, j))];
      }
    }
  }

  apply_u_velocity_bc(u_star_);
  apply_v_bc(v_star_);

  momentum_u_rhs_prev_ = u_adv;
  momentum_v_rhs_prev_ = v_adv;

  last_momentum_iterations_ = report.iterations;
  last_momentum_solver_name_ =
      "ExplicitConvection+MonolithicImplicitViscosityBiCGSTAB[" +
      std::string(momentum_preconditioner_name(preconditioner.type)) + "]";
  return report.absolute_residual;
}


} // namespace ding
