#include "solver.hpp"
#include "internal.hpp"

#include <algorithm>
#include <cmath>

namespace ding {

using coupled_detail::square;

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

double Solver::stress_divergence_u(const Field2D &u_state, const Field2D &v_state, const Field2D &eta_field, int i,
                                   int j) const {
  // u-control volume:
  // d/dx [ 2 eta du/dx ] + d/dy [ eta (du/dy + dv/dx) ]
  const double eta_e = eta_u_face(eta_field, i + 1, j);
  const double eta_w = eta_u_face(eta_field, i, j);
  const double dudx_e = (u_state(i + 1, j) - u_state(i, j)) / dx_;
  const double dudx_w = (u_state(i, j) - u_state(i - 1, j)) / dx_;
  const double normal = (2.0 * eta_e * dudx_e - 2.0 * eta_w * dudx_w) / dx_;

  const double eta_n = eta_corner(eta_field, i, j + 1);
  const double eta_s = eta_corner(eta_field, i, j);
  const double dudy_n = (u_state(i, j + 1) - u_state(i, j)) / dy_;
  const double dudy_s = (u_state(i, j) - u_state(i, j - 1)) / dy_;
  const double dvdx_n = (v_state(i, j + 1) - v_state(i - 1, j + 1)) / dx_;
  const double dvdx_s = (v_state(i, j) - v_state(i - 1, j)) / dx_;
  const double shear = (eta_n * (dudy_n + dvdx_n) - eta_s * (dudy_s + dvdx_s)) / dy_;

  return normal + shear;
}

double Solver::stress_divergence_v(const Field2D &u_state, const Field2D &v_state, int i, int j) const {
  return stress_divergence_v(u_state, v_state, eta_, i, j);
}

double Solver::stress_divergence_v(const Field2D &u_state, const Field2D &v_state, const Field2D &eta_field, int i,
                                   int j) const {
  // v-control volume:
  // d/dx [ eta (dv/dx + du/dy) ] + d/dy [ 2 eta dv/dy ]
  const double eta_e = eta_corner(eta_field, i + 1, j);
  const double eta_w = eta_corner(eta_field, i, j);
  const double dvdx_e = (v_state(i + 1, j) - v_state(i, j)) / dx_;
  const double dvdx_w = (v_state(i, j) - v_state(i - 1, j)) / dx_;
  const double dudy_e = (u_state(i + 1, j) - u_state(i + 1, j - 1)) / dy_;
  const double dudy_w = (u_state(i, j) - u_state(i, j - 1)) / dy_;
  const double shear = (eta_e * (dvdx_e + dudy_e) - eta_w * (dvdx_w + dudy_w)) / dx_;

  const double eta_n = eta_v_face(eta_field, i, j + 1);
  const double eta_s = eta_v_face(eta_field, i, j);
  const double dvdy_n = (v_state(i, j + 1) - v_state(i, j)) / dy_;
  const double dvdy_s = (v_state(i, j) - v_state(i, j - 1)) / dy_;
  const double normal = (2.0 * eta_n * dvdy_n - 2.0 * eta_s * dvdy_s) / dy_;

  return shear + normal;
}

double Solver::solve_momentum_predictor(const Field2D &u_adv, const Field2D &v_adv, int step) {
  u_star_ = u_;
  v_star_ = v_;
  double residual = 0.0;
  Field2D visc_old_u(u_.nx, u_.ny, u_.ghost, 0.0);
  Field2D visc_old_v(v_.nx, v_.ny, v_.ghost, 0.0);

  for (int i = 0; i < u_.nx; ++i) {
    for (int j = 0; j < u_.ny; ++j) {
      visc_old_u(i, j) = stress_divergence_u(u_, v_, eta_previous_step_, i, j) / cfg_.re;
    }
  }
  for (int i = 0; i < v_.nx; ++i) {
    for (int j = 0; j < v_.ny; ++j) {
      visc_old_v(i, j) = stress_divergence_v(u_, v_, eta_previous_step_, i, j) / cfg_.re;
    }
  }

  // Ding 2007 Eq. (27):
  // - convection uses Adams-Bashforth in time
  // - old pressure does not enter the predictor explicitly
  // - viscous stress uses Crank-Nicolson with eta^n in the old term and eta^{n+1} in the new term
  int iterations = 0;
  const double omega = 1.0;
  for (int iter = 0; iter < cfg_.momentum_iterations; ++iter) {
    double diff_sq = 0.0;
    double norm_sq = 0.0;
    Field2D u_next = u_star_;
    Field2D v_next = v_star_;

    for (int i = 0; i < u_.nx; ++i) {
      for (int j = 0; j < u_.ny; ++j) {
        const double rho_face = rho_u_face(rho_mid_, i, j);
        const double conv = (step == 0) ? u_adv(i, j) : 1.5 * u_adv(i, j) - 0.5 * momentum_u_rhs_prev_(i, j);
        const double visc_new = stress_divergence_u(u_star_, v_star_, eta_, i, j) / cfg_.re;
        const double explicit_rhs =
            u_(i, j) + cfg_.dt * (-conv + cfg_.body_force_x + surface_fx_u_(i, j) / rho_face) +
            0.5 * cfg_.dt * visc_old_u(i, j) / rho_face;
        const double eta_e = eta_u_face(eta_, i + 1, j);
        const double eta_w = eta_u_face(eta_, i, j);
        const double eta_n = eta_corner(eta_, i, j + 1);
        const double eta_s = eta_corner(eta_, i, j);
        const double diag_abs =
            2.0 * (eta_e + eta_w) / (dx_ * dx_) + (eta_n + eta_s) / (dy_ * dy_);
        const double diag = 1.0 + 0.5 * cfg_.dt * diag_abs / std::max(cfg_.re * rho_face, 1.0e-30);
        const double residual_local = explicit_rhs - (u_star_(i, j) - 0.5 * cfg_.dt * visc_new / rho_face);
        u_next(i, j) = u_star_(i, j) + omega * residual_local / std::max(diag, 1.0e-30);
        diff_sq += square(u_next(i, j) - u_star_(i, j));
        norm_sq += square(u_next(i, j));
      }
    }
    apply_u_bc(u_next);

    for (int i = 0; i < v_.nx; ++i) {
      for (int j = 0; j < v_.ny; ++j) {
        const double rho_face = rho_v_face(rho_mid_, i, j);
        const double conv = (step == 0) ? v_adv(i, j) : 1.5 * v_adv(i, j) - 0.5 * momentum_v_rhs_prev_(i, j);
        const double visc_new = stress_divergence_v(u_star_, v_star_, eta_, i, j) / cfg_.re;
        const double explicit_rhs =
            v_(i, j) + cfg_.dt * (-conv + cfg_.body_force_y + surface_fy_v_(i, j) / rho_face) +
            0.5 * cfg_.dt * visc_old_v(i, j) / rho_face;
        const double eta_e = eta_corner(eta_, i + 1, j);
        const double eta_w = eta_corner(eta_, i, j);
        const double eta_n = eta_v_face(eta_, i, j + 1);
        const double eta_s = eta_v_face(eta_, i, j);
        const double diag_abs =
            (eta_e + eta_w) / (dx_ * dx_) + 2.0 * (eta_n + eta_s) / (dy_ * dy_);
        const double diag = 1.0 + 0.5 * cfg_.dt * diag_abs / std::max(cfg_.re * rho_face, 1.0e-30);
        const double residual_local = explicit_rhs - (v_star_(i, j) - 0.5 * cfg_.dt * visc_new / rho_face);
        v_next(i, j) = v_star_(i, j) + omega * residual_local / std::max(diag, 1.0e-30);
        diff_sq += square(v_next(i, j) - v_star_(i, j));
        norm_sq += square(v_next(i, j));
      }
    }
    apply_v_bc(v_next);

    u_star_ = u_next;
    v_star_ = v_next;
    residual = std::sqrt(diff_sq / std::max(norm_sq, 1.0e-30));
    iterations = iter + 1;
    if (residual < cfg_.momentum_tolerance) {
      break;
    }
  }

  momentum_u_rhs_prev_ = u_adv;
  momentum_v_rhs_prev_ = v_adv;
  last_momentum_iterations_ = iterations;
  return residual;
}


} // namespace ding
