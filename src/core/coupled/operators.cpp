#include "solver.hpp"
#include "internal.hpp"

#include <algorithm>
#include <cmath>

namespace ding {

using coupled_detail::square;

void Solver::apply_scalar_bc(Field2D &field) const {
  for (int g = 1; g <= field.ghost; ++g) {
    for (int j = 0; j < field.ny; ++j) {
      field(-g, j) = cfg_.periodic_x ? field(field.nx - g, j) : field(0, j);
      field(field.nx - 1 + g, j) = cfg_.periodic_x ? field(g - 1, j) : field(field.nx - 1, j);
    }
    for (int i = -field.ghost; i < field.nx + field.ghost; ++i) {
      field(i, -g) = cfg_.periodic_y ? field(i, field.ny - g) : field(i, 0);
      field(i, field.ny - 1 + g) = cfg_.periodic_y ? field(i, g - 1) : field(i, field.ny - 1);
    }
  }
}

void Solver::apply_u_bc(Field2D &field) const {
  if (cfg_.periodic_x) {
    for (int j = 0; j < field.ny; ++j) {
      field(field.nx - 1, j) = field(0, j);
    }
  }
  for (int g = 1; g <= field.ghost; ++g) {
    for (int j = 0; j < field.ny; ++j) {
      field(-g, j) = cfg_.periodic_x ? field(field.nx - g - 1, j) : 0.0;
      field(field.nx - 1 + g, j) = cfg_.periodic_x ? field(g, j) : 0.0;
    }
    for (int i = -field.ghost; i < field.nx + field.ghost; ++i) {
      field(i, -g) = cfg_.periodic_y ? field(i, field.ny - g) : -field(i, g - 1);
      field(i, field.ny - 1 + g) = cfg_.periodic_y ? field(i, g - 1) : -field(i, field.ny - g);
    }
  }
}

void Solver::apply_u_velocity_bc(Field2D &field) const {
  apply_u_bc(field);
  if (cfg_.periodic_y) {
    return;
  }

  const double bottom_wall = cfg_.bottom_wall_velocity_x;
  const double top_wall = cfg_.top_wall_velocity_x;
  if (std::abs(bottom_wall) <= 0.0 && std::abs(top_wall) <= 0.0) {
    return;
  }

  for (int g = 1; g <= field.ghost; ++g) {
    for (int i = -field.ghost; i < field.nx + field.ghost; ++i) {
      field(i, -g) = 2.0 * bottom_wall - field(i, g - 1);
      field(i, field.ny - 1 + g) = 2.0 * top_wall - field(i, field.ny - g);
    }
  }
}

void Solver::apply_v_bc(Field2D &field) const {
  if (cfg_.periodic_y) {
    for (int i = 0; i < field.nx; ++i) {
      field(i, field.ny - 1) = field(i, 0);
    }
  }
  for (int g = 1; g <= field.ghost; ++g) {
    for (int j = 0; j < field.ny; ++j) {
      field(-g, j) = cfg_.periodic_x ? field(field.nx - g, j) : -field(g - 1, j);
      field(field.nx - 1 + g, j) = cfg_.periodic_x ? field(g - 1, j) : -field(field.nx - g, j);
    }
    for (int i = -field.ghost; i < field.nx + field.ghost; ++i) {
      field(i, -g) = cfg_.periodic_y ? field(i, field.ny - g - 1) : 0.0;
      field(i, field.ny - 1 + g) = cfg_.periodic_y ? field(i, g) : 0.0;
    }
  }
}

double Solver::clamp_phase_debug(double value) const {
  if (!cfg_.use_phase_clamp_debug_only) {
    return value;
  }
  return std::clamp(value, -0.05, 1.05);
}


double Solver::laplacian_center(const Field2D &field, int i, int j) const {
  return (field(i + 1, j) - 2.0 * field(i, j) + field(i - 1, j)) / (dx_ * dx_) +
         (field(i, j + 1) - 2.0 * field(i, j) + field(i, j - 1)) / (dy_ * dy_);
}

double Solver::grad_center_x(const Field2D &field, int i, int j) const {
  return (field(i + 1, j) - field(i - 1, j)) / (2.0 * dx_);
}

double Solver::grad_center_y(const Field2D &field, int i, int j) const {
  return (field(i, j + 1) - field(i, j - 1)) / (2.0 * dy_);
}

double Solver::grad_face_x_centered(const Field2D &field, int i_face, int j) const {
  return (field(i_face, j) - field(i_face - 1, j)) / dx_;
}

double Solver::grad_face_y_centered(const Field2D &field, int i, int j_face) const {
  return (field(i, j_face) - field(i, j_face - 1)) / dy_;
}

double Solver::cell_centered_u(const Field2D &u_state, int i, int j) const {
  return 0.5 * (u_state(i, j) + u_state(i + 1, j));
}

double Solver::cell_centered_v(const Field2D &v_state, int i, int j) const {
  return 0.5 * (v_state(i, j) + v_state(i, j + 1));
}

double Solver::rho_u_face(int i, int j) const { return rho_u_face(rho_, i, j); }

double Solver::rho_u_face(const Field2D &rho_field, int i, int j) const {
  if (i <= 0) {
    return cfg_.periodic_x ? 0.5 * (rho_field(cfg_.nx - 1, j) + rho_field(0, j)) : rho_field(0, j);
  }
  if (i >= cfg_.nx) {
    return cfg_.periodic_x ? 0.5 * (rho_field(cfg_.nx - 1, j) + rho_field(0, j)) : rho_field(cfg_.nx - 1, j);
  }
  return 0.5 * (rho_field(i - 1, j) + rho_field(i, j));
}

double Solver::rho_v_face(int i, int j) const { return rho_v_face(rho_, i, j); }

double Solver::rho_v_face(const Field2D &rho_field, int i, int j) const {
  if (j <= 0) {
    return cfg_.periodic_y ? 0.5 * (rho_field(i, cfg_.ny - 1) + rho_field(i, 0)) : rho_field(i, 0);
  }
  if (j >= cfg_.ny) {
    return cfg_.periodic_y ? 0.5 * (rho_field(i, cfg_.ny - 1) + rho_field(i, 0)) : rho_field(i, cfg_.ny - 1);
  }
  return 0.5 * (rho_field(i, j - 1) + rho_field(i, j));
}

double Solver::eta_u_face(int i, int j) const { return eta_u_face(eta_, i, j); }

double Solver::eta_u_face(const Field2D &eta_field, int i, int j) const {
  if (i <= 0) {
    return cfg_.periodic_x ? 0.5 * (eta_field(cfg_.nx - 1, j) + eta_field(0, j)) : eta_field(0, j);
  }
  if (i >= cfg_.nx) {
    return cfg_.periodic_x ? 0.5 * (eta_field(cfg_.nx - 1, j) + eta_field(0, j)) : eta_field(cfg_.nx - 1, j);
  }
  return 0.5 * (eta_field(i - 1, j) + eta_field(i, j));
}

double Solver::eta_v_face(int i, int j) const { return eta_v_face(eta_, i, j); }

double Solver::eta_v_face(const Field2D &eta_field, int i, int j) const {
  if (j <= 0) {
    return cfg_.periodic_y ? 0.5 * (eta_field(i, cfg_.ny - 1) + eta_field(i, 0)) : eta_field(i, 0);
  }
  if (j >= cfg_.ny) {
    return cfg_.periodic_y ? 0.5 * (eta_field(i, cfg_.ny - 1) + eta_field(i, 0)) : eta_field(i, cfg_.ny - 1);
  }
  return 0.5 * (eta_field(i, j - 1) + eta_field(i, j));
}

double Solver::rho_corner(int i, int j) const { return rho_corner(rho_, i, j); }

double Solver::rho_corner(const Field2D &rho_field, int i, int j) const {
  return 0.25 * (rho_field(i, j) + rho_field(i - 1, j) + rho_field(i, j - 1) + rho_field(i - 1, j - 1));
}

double Solver::eta_corner(int i, int j) const { return eta_corner(eta_, i, j); }

double Solver::eta_corner(const Field2D &eta_field, int i, int j) const {
  return 0.25 * (eta_field(i, j) + eta_field(i - 1, j) + eta_field(i, j - 1) + eta_field(i - 1, j - 1));
}

double Solver::divergence_cell(const Field2D &u_state, const Field2D &v_state, int i, int j) const {
  return (u_state(i + 1, j) - u_state(i, j)) / dx_ + (v_state(i, j + 1) - v_state(i, j)) / dy_;
}

double Solver::pressure_gradient_u_face(const Field2D &pressure_like, int i, int j) const {
  if (!cfg_.periodic_x && (i <= 0 || i >= cfg_.nx)) {
    return 0.0;
  }
  return (pressure_like(i, j) - pressure_like(i - 1, j)) / dx_;
}

double Solver::pressure_gradient_v_face(const Field2D &pressure_like, int i, int j) const {
  if (!cfg_.periodic_y && (j <= 0 || j >= cfg_.ny)) {
    return 0.0;
  }
  return (pressure_like(i, j) - pressure_like(i, j - 1)) / dy_;
}

bool Solver::use_liu_pressure_split() const {
  return cfg_.pressure_scheme == "liu_split_icpcg" || cfg_.pressure_scheme == "split_icpcg" ||
         cfg_.pressure_scheme == "paper_split_icpcg" || cfg_.pressure_scheme == "liu_split_ildlt_pcg" ||
         cfg_.pressure_scheme == "split_ildlt_pcg" || cfg_.pressure_scheme == "paper_split_ildlt_pcg";
}

double Solver::liu_split_reference_density() const { return std::min(1.0, cfg_.density_ratio); }

void Solver::build_liu_split_pressure_extrapolation(Field2D &pressure_extrapolated) const {
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      pressure_extrapolated(i, j) = (current_step_index_ == 0)
                                        ? pressure_(i, j)
                                        : (2.0 * pressure_(i, j) - pressure_previous_step_(i, j));
    }
  }
  apply_scalar_bc(pressure_extrapolated);
}

double Solver::liu_split_explicit_divergence(const Field2D &pressure_extrapolated, int i, int j) const {
  const double rho_ref = liu_split_reference_density();
  const bool east_open = cfg_.periodic_x || i < cfg_.nx - 1;
  const bool west_open = cfg_.periodic_x || i > 0;
  const bool north_open = cfg_.periodic_y || j < cfg_.ny - 1;
  const bool south_open = cfg_.periodic_y || j > 0;

  const double flux_e = east_open ? (1.0 - rho_ref / rho_u_face(rho_mid_, i + 1, j)) *
                                        grad_face_x_centered(pressure_extrapolated, i + 1, j)
                                  : 0.0;
  const double flux_w = west_open ? (1.0 - rho_ref / rho_u_face(rho_mid_, i, j)) *
                                        grad_face_x_centered(pressure_extrapolated, i, j)
                                  : 0.0;
  const double flux_n = north_open ? (1.0 - rho_ref / rho_v_face(rho_mid_, i, j + 1)) *
                                         grad_face_y_centered(pressure_extrapolated, i, j + 1)
                                   : 0.0;
  const double flux_s = south_open ? (1.0 - rho_ref / rho_v_face(rho_mid_, i, j)) *
                                         grad_face_y_centered(pressure_extrapolated, i, j)
                                   : 0.0;
  return (flux_e - flux_w) / dx_ + (flux_n - flux_s) / dy_;
}


void Solver::subtract_mean(Field2D &field) const {
  double mean = 0.0;
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      mean += field(i, j);
    }
  }
  mean /= static_cast<double>(cfg_.nx * cfg_.ny);
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      field(i, j) -= mean;
    }
  }
}


double Solver::periodic_boundary_speed_stat(const Field2D &u_state, const Field2D &v_state) const {
  double max_speed = 0.0;
  if (cfg_.periodic_x) {
    for (int j = 0; j < u_state.ny; ++j) {
      max_speed = std::max(max_speed, std::abs(u_state(0, j)));
      max_speed = std::max(max_speed, std::abs(u_state(cfg_.nx, j)));
    }
  }
  if (cfg_.periodic_y) {
    for (int i = 0; i < v_state.nx; ++i) {
      max_speed = std::max(max_speed, std::abs(v_state(i, 0)));
      max_speed = std::max(max_speed, std::abs(v_state(i, cfg_.ny)));
    }
  }
  return max_speed;
}

void Solver::apply_pressure_velocity_correction() {
  last_boundary_speed_pre_correction_ = periodic_boundary_speed_stat(u_star_, v_star_);

  Field2D pressure_extrapolated(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0);
  const bool liu_split = use_liu_pressure_split();
  const double rho_ref = liu_split ? liu_split_reference_density() : 1.0;
  if (liu_split) {
    build_liu_split_pressure_extrapolation(pressure_extrapolated);
  }

  for (int i = 0; i < u_.nx; ++i) {
    for (int j = 0; j < u_.ny; ++j) {
      if (!cfg_.periodic_x && (i == 0 || i == cfg_.nx)) {
        u_(i, j) = 0.0;
      } else {
        // Liu et al. (2021), Eq. (26):
        // u^{n+1} = u* - dt [ (1/rho_ref) grad(p^{n+1/2}) + (1/rho_mid - 1/rho_ref) grad(p_ext) ]
        const double split_explicit = liu_split
                                          ? (1.0 / rho_u_face(rho_mid_, i, j) - 1.0 / rho_ref) *
                                                pressure_gradient_u_face(pressure_extrapolated, i, j)
                                          : 0.0;
        const double split_implicit =
            liu_split ? pressure_gradient_u_face(pressure_correction_, i, j) / rho_ref
                      : pressure_gradient_u_face(pressure_correction_, i, j) / rho_u_face(rho_mid_, i, j);
        u_(i, j) = u_star_(i, j) - cfg_.dt * (split_implicit + split_explicit);
      }
    }
  }

  for (int i = 0; i < v_.nx; ++i) {
    for (int j = 0; j < v_.ny; ++j) {
      if (!cfg_.periodic_y && (j == 0 || j == cfg_.ny)) {
        v_(i, j) = 0.0;
      } else {
        const double split_explicit = liu_split
                                          ? (1.0 / rho_v_face(rho_mid_, i, j) - 1.0 / rho_ref) *
                                                pressure_gradient_v_face(pressure_extrapolated, i, j)
                                          : 0.0;
        const double split_implicit =
            liu_split ? pressure_gradient_v_face(pressure_correction_, i, j) / rho_ref
                      : pressure_gradient_v_face(pressure_correction_, i, j) / rho_v_face(rho_mid_, i, j);
        v_(i, j) = v_star_(i, j) - cfg_.dt * (split_implicit + split_explicit);
      }
    }
  }
  apply_u_velocity_bc(u_);
  apply_v_bc(v_);
  last_boundary_speed_post_correction_ = periodic_boundary_speed_stat(u_, v_);

  // Ding 2007 Eq. (28)-(29):
  // pressure_correction_ here stores the projected pressure p^{n+1/2},
  // not an incremental delta_p.
  pressure_previous_step_ = pressure_;
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      pressure_(i, j) = pressure_correction_(i, j);
    }
  }
  subtract_mean(pressure_);
  apply_scalar_bc(pressure_);
}

double Solver::field_diff_l2(const Field2D &a, const Field2D &b, int i_begin, int i_end, int j_begin, int j_end) const {
  double diff_sq = 0.0;
  double norm_sq = 0.0;
  for (int i = i_begin; i < i_end; ++i) {
    for (int j = j_begin; j < j_end; ++j) {
      diff_sq += square(a(i, j) - b(i, j));
      norm_sq += square(a(i, j));
    }
  }
  return std::sqrt(diff_sq / std::max(norm_sq, 1.0e-30));
}

double Solver::compute_coupling_residuals(const Field2D &c_old, const Field2D &u_old, const Field2D &v_old,
                                          const Field2D &p_old) const {
  const double rc = field_diff_l2(c_, c_old, 0, cfg_.nx, 0, cfg_.ny);
  const double ru = field_diff_l2(u_, u_old, 0, u_.nx, 0, u_.ny);
  const double rv = field_diff_l2(v_, v_old, 0, v_.nx, 0, v_.ny);
  const double rp = field_diff_l2(pressure_, p_old, 0, cfg_.nx, 0, cfg_.ny);
  return std::max(std::max(rc, ru), std::max(rv, rp));
}


} // namespace ding
