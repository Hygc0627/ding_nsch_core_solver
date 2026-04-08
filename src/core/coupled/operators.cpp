#include "solver.hpp"
#include "internal.hpp"

#include <algorithm>
#include <cmath>

namespace ding {

using coupled_detail::square;

void Solver::validate_boundary_configuration() const {
  auto require_no_explicit_bc = [&](const BoundaryConditionSpec &bc, const char *name) {
    if (bc.type != BoundaryConditionType::unset) {
      throw std::runtime_error(std::string(name) + " cannot be set when the direction is periodic");
    }
  };

  if (cfg_.periodic_x) {
    require_no_explicit_bc(cfg_.pressure_bc_left, "pressure_bc_left");
    require_no_explicit_bc(cfg_.pressure_bc_right, "pressure_bc_right");
    require_no_explicit_bc(cfg_.u_bc_left, "u_bc_left");
    require_no_explicit_bc(cfg_.u_bc_right, "u_bc_right");
    require_no_explicit_bc(cfg_.v_bc_left, "v_bc_left");
    require_no_explicit_bc(cfg_.v_bc_right, "v_bc_right");
  }
  if (cfg_.periodic_y) {
    require_no_explicit_bc(cfg_.pressure_bc_bottom, "pressure_bc_bottom");
    require_no_explicit_bc(cfg_.pressure_bc_top, "pressure_bc_top");
    require_no_explicit_bc(cfg_.u_bc_bottom, "u_bc_bottom");
    require_no_explicit_bc(cfg_.u_bc_top, "u_bc_top");
    require_no_explicit_bc(cfg_.v_bc_bottom, "v_bc_bottom");
    require_no_explicit_bc(cfg_.v_bc_top, "v_bc_top");
  }
}

BoundaryConditionSpec Solver::effective_pressure_bc(BoundarySide side) const {
  switch (side) {
  case BoundarySide::left:
    if (cfg_.pressure_bc_left.type != BoundaryConditionType::unset) {
      return cfg_.pressure_bc_left;
    }
    break;
  case BoundarySide::right:
    if (cfg_.pressure_bc_right.type != BoundaryConditionType::unset) {
      return cfg_.pressure_bc_right;
    }
    break;
  case BoundarySide::bottom:
    if (cfg_.pressure_bc_bottom.type != BoundaryConditionType::unset) {
      return cfg_.pressure_bc_bottom;
    }
    break;
  case BoundarySide::top:
    if (cfg_.pressure_bc_top.type != BoundaryConditionType::unset) {
      return cfg_.pressure_bc_top;
    }
    break;
  }
  return {BoundaryConditionType::neumann, 0.0};
}

BoundaryConditionSpec Solver::effective_u_bc(BoundarySide side) const {
  switch (side) {
  case BoundarySide::left:
    if (cfg_.u_bc_left.type != BoundaryConditionType::unset) {
      return cfg_.u_bc_left;
    }
    return {BoundaryConditionType::dirichlet, 0.0};
  case BoundarySide::right:
    if (cfg_.u_bc_right.type != BoundaryConditionType::unset) {
      return cfg_.u_bc_right;
    }
    return {BoundaryConditionType::dirichlet, 0.0};
  case BoundarySide::bottom:
    if (cfg_.u_bc_bottom.type != BoundaryConditionType::unset) {
      return cfg_.u_bc_bottom;
    }
    return {BoundaryConditionType::dirichlet, cfg_.bottom_wall_velocity_x};
  case BoundarySide::top:
    if (cfg_.u_bc_top.type != BoundaryConditionType::unset) {
      return cfg_.u_bc_top;
    }
    return {BoundaryConditionType::dirichlet, cfg_.top_wall_velocity_x};
  }
  return {BoundaryConditionType::dirichlet, 0.0};
}

BoundaryConditionSpec Solver::effective_v_bc(BoundarySide side) const {
  switch (side) {
  case BoundarySide::left:
    if (cfg_.v_bc_left.type != BoundaryConditionType::unset) {
      return cfg_.v_bc_left;
    }
    break;
  case BoundarySide::right:
    if (cfg_.v_bc_right.type != BoundaryConditionType::unset) {
      return cfg_.v_bc_right;
    }
    break;
  case BoundarySide::bottom:
    if (cfg_.v_bc_bottom.type != BoundaryConditionType::unset) {
      return cfg_.v_bc_bottom;
    }
    break;
  case BoundarySide::top:
    if (cfg_.v_bc_top.type != BoundaryConditionType::unset) {
      return cfg_.v_bc_top;
    }
    break;
  }
  return {BoundaryConditionType::dirichlet, 0.0};
}

bool Solver::pressure_has_dirichlet_boundary() const {
  if (!cfg_.periodic_x) {
    if (effective_pressure_bc(BoundarySide::left).type == BoundaryConditionType::dirichlet ||
        effective_pressure_bc(BoundarySide::right).type == BoundaryConditionType::dirichlet) {
      return true;
    }
  }
  if (!cfg_.periodic_y) {
    if (effective_pressure_bc(BoundarySide::bottom).type == BoundaryConditionType::dirichlet ||
        effective_pressure_bc(BoundarySide::top).type == BoundaryConditionType::dirichlet) {
      return true;
    }
  }
  return false;
}

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

void Solver::apply_pressure_bc(Field2D &field) const {
  const BoundaryConditionSpec left_bc = effective_pressure_bc(BoundarySide::left);
  const BoundaryConditionSpec right_bc = effective_pressure_bc(BoundarySide::right);
  const BoundaryConditionSpec bottom_bc = effective_pressure_bc(BoundarySide::bottom);
  const BoundaryConditionSpec top_bc = effective_pressure_bc(BoundarySide::top);

  for (int g = 1; g <= field.ghost; ++g) {
    for (int j = 0; j < field.ny; ++j) {
      if (cfg_.periodic_x) {
        field(-g, j) = field(field.nx - g, j);
        field(field.nx - 1 + g, j) = field(g - 1, j);
      } else {
        const double left_offset = 2.0 * (static_cast<double>(g) - 0.5) * left_bc.value * dx_;
        const double right_offset = 2.0 * (static_cast<double>(g) - 0.5) * right_bc.value * dx_;
        field(-g, j) = (left_bc.type == BoundaryConditionType::dirichlet)
                           ? 2.0 * left_bc.value - field(g - 1, j)
                           : field(g - 1, j) + left_offset;
        field(field.nx - 1 + g, j) = (right_bc.type == BoundaryConditionType::dirichlet)
                                         ? 2.0 * right_bc.value - field(field.nx - g, j)
                                         : field(field.nx - g, j) + right_offset;
      }
    }
    for (int i = -field.ghost; i < field.nx + field.ghost; ++i) {
      if (cfg_.periodic_y) {
        field(i, -g) = field(i, field.ny - g);
        field(i, field.ny - 1 + g) = field(i, g - 1);
      } else {
        const double bottom_offset = 2.0 * (static_cast<double>(g) - 0.5) * bottom_bc.value * dy_;
        const double top_offset = 2.0 * (static_cast<double>(g) - 0.5) * top_bc.value * dy_;
        field(i, -g) = (bottom_bc.type == BoundaryConditionType::dirichlet)
                           ? 2.0 * bottom_bc.value - field(i, g - 1)
                           : field(i, g - 1) + bottom_offset;
        field(i, field.ny - 1 + g) = (top_bc.type == BoundaryConditionType::dirichlet)
                                         ? 2.0 * top_bc.value - field(i, field.ny - g)
                                         : field(i, field.ny - g) + top_offset;
      }
    }
  }
}

void Solver::apply_u_bc(Field2D &field) const {
  const BoundaryConditionSpec left_bc = effective_u_bc(BoundarySide::left);
  const BoundaryConditionSpec right_bc = effective_u_bc(BoundarySide::right);
  const BoundaryConditionSpec bottom_bc = effective_u_bc(BoundarySide::bottom);
  const BoundaryConditionSpec top_bc = effective_u_bc(BoundarySide::top);

  if (cfg_.periodic_x) {
    for (int j = 0; j < field.ny; ++j) {
      field(field.nx - 1, j) = field(0, j);
    }
  } else {
    for (int j = 0; j < field.ny; ++j) {
      field(0, j) = (left_bc.type == BoundaryConditionType::dirichlet) ? left_bc.value : field(1, j) + left_bc.value * dx_;
      field(field.nx - 1, j) = (right_bc.type == BoundaryConditionType::dirichlet)
                                   ? right_bc.value
                                   : field(field.nx - 2, j) + right_bc.value * dx_;
    }
  }
  for (int g = 1; g <= field.ghost; ++g) {
    for (int j = 0; j < field.ny; ++j) {
      if (cfg_.periodic_x) {
        field(-g, j) = field(field.nx - g - 1, j);
        field(field.nx - 1 + g, j) = field(g, j);
      } else {
        const double left_offset = 2.0 * static_cast<double>(g) * left_bc.value * dx_;
        const double right_offset = 2.0 * static_cast<double>(g) * right_bc.value * dx_;
        field(-g, j) = (left_bc.type == BoundaryConditionType::dirichlet)
                           ? 2.0 * left_bc.value - field(g, j)
                           : field(g, j) + left_offset;
        field(field.nx - 1 + g, j) = (right_bc.type == BoundaryConditionType::dirichlet)
                                         ? 2.0 * right_bc.value - field(field.nx - 1 - g, j)
                                         : field(field.nx - 1 - g, j) + right_offset;
      }
    }
    for (int i = -field.ghost; i < field.nx + field.ghost; ++i) {
      if (cfg_.periodic_y) {
        field(i, -g) = field(i, field.ny - g);
        field(i, field.ny - 1 + g) = field(i, g - 1);
      } else {
        const double bottom_offset = 2.0 * (static_cast<double>(g) - 0.5) * bottom_bc.value * dy_;
        const double top_offset = 2.0 * (static_cast<double>(g) - 0.5) * top_bc.value * dy_;
        field(i, -g) = (bottom_bc.type == BoundaryConditionType::dirichlet)
                           ? 2.0 * bottom_bc.value - field(i, g - 1)
                           : field(i, g - 1) + bottom_offset;
        field(i, field.ny - 1 + g) = (top_bc.type == BoundaryConditionType::dirichlet)
                                         ? 2.0 * top_bc.value - field(i, field.ny - g)
                                         : field(i, field.ny - g) + top_offset;
      }
    }
  }
}

void Solver::apply_u_velocity_bc(Field2D &field) const {
  apply_u_bc(field);
}

void Solver::apply_v_bc(Field2D &field) const {
  const BoundaryConditionSpec left_bc = effective_v_bc(BoundarySide::left);
  const BoundaryConditionSpec right_bc = effective_v_bc(BoundarySide::right);
  const BoundaryConditionSpec bottom_bc = effective_v_bc(BoundarySide::bottom);
  const BoundaryConditionSpec top_bc = effective_v_bc(BoundarySide::top);

  if (cfg_.periodic_y) {
    for (int i = 0; i < field.nx; ++i) {
      field(i, field.ny - 1) = field(i, 0);
    }
  } else {
    for (int i = 0; i < field.nx; ++i) {
      field(i, 0) = (bottom_bc.type == BoundaryConditionType::dirichlet) ? bottom_bc.value : field(i, 1) + bottom_bc.value * dy_;
      field(i, field.ny - 1) = (top_bc.type == BoundaryConditionType::dirichlet)
                                   ? top_bc.value
                                   : field(i, field.ny - 2) + top_bc.value * dy_;
    }
  }
  for (int g = 1; g <= field.ghost; ++g) {
    for (int j = 0; j < field.ny; ++j) {
      if (cfg_.periodic_x) {
        field(-g, j) = field(field.nx - g, j);
        field(field.nx - 1 + g, j) = field(g - 1, j);
      } else {
        const double left_offset = 2.0 * (static_cast<double>(g) - 0.5) * left_bc.value * dx_;
        const double right_offset = 2.0 * (static_cast<double>(g) - 0.5) * right_bc.value * dx_;
        field(-g, j) = (left_bc.type == BoundaryConditionType::dirichlet)
                           ? 2.0 * left_bc.value - field(g - 1, j)
                           : field(g - 1, j) + left_offset;
        field(field.nx - 1 + g, j) = (right_bc.type == BoundaryConditionType::dirichlet)
                                         ? 2.0 * right_bc.value - field(field.nx - g, j)
                                         : field(field.nx - g, j) + right_offset;
      }
    }
    for (int i = -field.ghost; i < field.nx + field.ghost; ++i) {
      if (cfg_.periodic_y) {
        field(i, -g) = field(i, field.ny - g - 1);
        field(i, field.ny - 1 + g) = field(i, g);
      } else {
        const double bottom_offset = 2.0 * static_cast<double>(g) * bottom_bc.value * dy_;
        const double top_offset = 2.0 * static_cast<double>(g) * top_bc.value * dy_;
        field(i, -g) = (bottom_bc.type == BoundaryConditionType::dirichlet)
                           ? 2.0 * bottom_bc.value - field(i, g)
                           : field(i, g) + bottom_offset;
        field(i, field.ny - 1 + g) = (top_bc.type == BoundaryConditionType::dirichlet)
                                         ? 2.0 * top_bc.value - field(i, field.ny - 1 - g)
                                         : field(i, field.ny - 1 - g) + top_offset;
      }
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
  return (pressure_like(i, j) - pressure_like(i - 1, j)) / dx_;
}

double Solver::pressure_gradient_v_face(const Field2D &pressure_like, int i, int j) const {
  return (pressure_like(i, j) - pressure_like(i, j - 1)) / dy_;
}

bool Solver::use_liu_pressure_split() const {
  return cfg_.pressure_scheme == "liu_split_icpcg" || cfg_.pressure_scheme == "split_icpcg" ||
         cfg_.pressure_scheme == "paper_split_icpcg" || cfg_.pressure_scheme == "liu_split_ildlt_pcg" ||
         cfg_.pressure_scheme == "split_ildlt_pcg" || cfg_.pressure_scheme == "paper_split_ildlt_pcg" ||
         cfg_.pressure_scheme == "split_petsc_pcg" || cfg_.pressure_scheme == "paper_split_petsc_pcg" ||
         cfg_.pressure_scheme == "liu_split_petsc_pcg" || cfg_.pressure_scheme == "split_hydea" ||
         cfg_.pressure_scheme == "paper_split_hydea" || cfg_.pressure_scheme == "liu_split_hydea";
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
  apply_pressure_bc(pressure_extrapolated);
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

  for (int i = 0; i < v_.nx; ++i) {
    for (int j = 0; j < v_.ny; ++j) {
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
  if (!pressure_has_dirichlet_boundary()) {
    subtract_mean(pressure_);
  }
  apply_pressure_bc(pressure_);
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
