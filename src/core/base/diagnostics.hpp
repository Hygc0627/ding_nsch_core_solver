#pragma once

#include <string>
#include <vector>

namespace ding {

struct Diagnostics {
  double mass = 0.0;
  double mass_drift = 0.0;
  double divergence_l2 = 0.0;
  double max_divergence_after_correction = 0.0;
  double max_abs_mu = 0.0;
  double max_velocity = 0.0;
  double kinetic_energy = 0.0;
  double total_free_energy = 0.0;
  double ch_inner_residual = 0.0;
  double ch_equation_residual = 0.0;
  double coupling_residual = 0.0;
  double pressure_correction_residual = 0.0;
  double momentum_residual = 0.0;
  double boundary_speed_pre_correction = 0.0;
  double boundary_speed_post_correction = 0.0;
  double rho_min = 0.0;
  double rho_max = 0.0;
  double eta_min = 0.0;
  double eta_max = 0.0;
  double dt_limit_advective = 0.0;
  double dt_limit_capillary = 0.0;
  double dt_limit_ch_explicit = 0.0;
  double dt_limit_active = 0.0;
  double dt_limit_ratio = 0.0;
  int ch_iterations = 0;
  int coupling_iterations = 0;
  int pressure_iterations = 0;
  int momentum_iterations = 0;
  std::string ch_solver_name;
  std::string momentum_solver_name;
  std::string pressure_solver_name;
  std::string dt_limit_source;
};

struct HistoryEntry {
  int step = 0;
  double time = 0.0;
  Diagnostics diag{};
};

} // namespace ding
