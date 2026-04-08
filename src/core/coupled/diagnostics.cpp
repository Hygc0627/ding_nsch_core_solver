#include "solver.hpp"
#include "internal.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

namespace ding {

using coupled_detail::square;

namespace {

constexpr double kAdvectiveSafety = 0.5;
constexpr double kCapillarySafety = 0.5;
constexpr double kChExplicitSafety = 0.25;
constexpr double kTiny = 1.0e-30;

} // namespace

double Solver::compute_mass(const Field2D &field) const {
  double sum = 0.0;
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      sum += field(i, j) * dx_ * dy_;
    }
  }
  return sum;
}

double Solver::compute_free_energy() const {
  double energy = 0.0;
  const double alpha = 6.0 * std::sqrt(2.0);
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      const double c = c_(i, j);
      const double bulk = 0.25 * c * c * square(1.0 - c) / cfg_.cn;
      const double grad_sq = square(grad_center_x(c_, i, j)) + square(grad_center_y(c_, i, j));
      energy += alpha * (bulk + 0.5 * cfg_.cn * grad_sq) * dx_ * dy_;
    }
  }
  return energy;
}

void Solver::populate_timestep_limits(Diagnostics &diag) const {
  const double h = std::min(dx_, dy_);
  double max_advective_rate = 0.0;
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      const double uc = cell_centered_u(u_, i, j);
      const double vc = cell_centered_v(v_, i, j);
      max_advective_rate = std::max(max_advective_rate, std::abs(uc) / std::max(dx_, kTiny) +
                                                            std::abs(vc) / std::max(dy_, kTiny));
    }
  }
  diag.dt_limit_advective =
      max_advective_rate > kTiny ? kAdvectiveSafety / max_advective_rate : std::numeric_limits<double>::infinity();

  diag.dt_limit_capillary = std::numeric_limits<double>::infinity();
  diag.dt_limit_ch_explicit = std::numeric_limits<double>::infinity();
  if (!is_advection_only_mode() && !is_single_phase_mode()) {
    const double rho_scale = std::max(diag.rho_max, kTiny);
    diag.dt_limit_capillary =
        kCapillarySafety * std::sqrt(std::max(cfg_.re * cfg_.ca * h * h * h / rho_scale, 0.0));

    const double alpha = 6.0 * std::sqrt(2.0);
    double mobility_max = 0.0;
    double max_abs_wpp = 0.0;
    for (int i = 0; i < cfg_.nx; ++i) {
      for (int j = 0; j < cfg_.ny; ++j) {
        const double c = c_(i, j);
        mobility_max = std::max(mobility_max, std::max(0.0, c * (1.0 - c)));
        const double wpp = 0.5 - 3.0 * c + 3.0 * c * c;
        max_abs_wpp = std::max(max_abs_wpp, std::abs(wpp));
      }
    }

    if (mobility_max > kTiny) {
      // Ding et al. (2007) use the split semi-implicit CH discretization of their
      // Eq. (25), specifically to remove the explicit timestep restriction from the
      // fourth-order diffusion term. For a paper-consistent diagnostic, do not treat
      // the biharmonic stiffness as an explicit h^4-limited process here.
      //
      // The remaining explicit stiffness comes primarily from the local bulk-energy
      // curvature in the nonlinear mobility-weighted term A(C,u), which behaves like
      // a second-order diffusion with coefficient ~ alpha * M * W'' / (Pe * Cn).
      const double coeff_lap = alpha * mobility_max * max_abs_wpp / std::max(cfg_.pe * cfg_.cn, kTiny);
      diag.dt_limit_ch_explicit =
          coeff_lap > kTiny ? kChExplicitSafety * h * h / coeff_lap : std::numeric_limits<double>::infinity();
    }
  }

  diag.dt_limit_active = diag.dt_limit_advective;
  diag.dt_limit_source = "advective";
  if (diag.dt_limit_capillary < diag.dt_limit_active) {
    diag.dt_limit_active = diag.dt_limit_capillary;
    diag.dt_limit_source = "capillary";
  }
  if (diag.dt_limit_ch_explicit < diag.dt_limit_active) {
    diag.dt_limit_active = diag.dt_limit_ch_explicit;
    diag.dt_limit_source = "ch_explicit";
  }
  if (!std::isfinite(diag.dt_limit_active) || diag.dt_limit_active <= kTiny) {
    diag.dt_limit_ratio = 0.0;
    diag.dt_limit_source = "unbounded";
  } else {
    diag.dt_limit_ratio = cfg_.dt / diag.dt_limit_active;
  }
}

Diagnostics Solver::compute_diagnostics() const {
  Diagnostics diag;
  diag.mass = compute_mass(c_);
  diag.mass_drift = std::abs(diag.mass - initial_mass_);

  double div_sq = 0.0;
  double div_max = 0.0;
  double mu_max = 0.0;
  double vel_max = 0.0;
  double ke = 0.0;
  double rho_min = std::numeric_limits<double>::infinity();
  double rho_max = 0.0;
  double eta_min = std::numeric_limits<double>::infinity();
  double eta_max = 0.0;
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      const double div = divergence_cell(u_, v_, i, j);
      const double uc = cell_centered_u(u_, i, j);
      const double vc = cell_centered_v(v_, i, j);
      div_sq += div * div;
      div_max = std::max(div_max, std::abs(div));
      mu_max = std::max(mu_max, std::abs(mu_(i, j)));
      vel_max = std::max(vel_max, std::sqrt(uc * uc + vc * vc));
      ke += 0.5 * rho_(i, j) * (uc * uc + vc * vc) * dx_ * dy_;
      rho_min = std::min(rho_min, rho_(i, j));
      rho_max = std::max(rho_max, rho_(i, j));
      eta_min = std::min(eta_min, eta_(i, j));
      eta_max = std::max(eta_max, eta_(i, j));
    }
  }

  diag.divergence_l2 = std::sqrt(div_sq / static_cast<double>(cfg_.nx * cfg_.ny));
  diag.max_divergence_after_correction = div_max;
  diag.max_abs_mu = mu_max;
  diag.max_velocity = vel_max;
  diag.kinetic_energy = ke;
  diag.total_free_energy = compute_free_energy();
  diag.ch_equation_residual = last_ch_equation_residual_;
  diag.boundary_speed_pre_correction = last_boundary_speed_pre_correction_;
  diag.boundary_speed_post_correction = last_boundary_speed_post_correction_;
  diag.rho_min = rho_min;
  diag.rho_max = rho_max;
  diag.eta_min = eta_min;
  diag.eta_max = eta_max;
  populate_timestep_limits(diag);
  diag.ch_iterations = last_ch_iterations_;
  diag.coupling_iterations = last_coupling_iterations_;
  diag.pressure_iterations = last_pressure_iterations_;
  diag.momentum_iterations = last_momentum_iterations_;
  diag.ch_solver_name = last_ch_solver_name_;
  diag.momentum_solver_name = last_momentum_solver_name_;
  diag.pressure_solver_name = last_pressure_solver_name_;
  return diag;
}

std::string Solver::format_step_report(int step, double time, const Diagnostics &diag) const {
  std::ostringstream out;
  out << std::fixed << std::setprecision(6)
      << "[step " << step << "] time=" << time << " dt=" << cfg_.dt
      << " mass=" << diag.mass << " mass_drift=" << diag.mass_drift
      << " div_l2=" << diag.divergence_l2 << " div_max=" << diag.max_divergence_after_correction
      << " max|mu|=" << diag.max_abs_mu << " max|u|=" << diag.max_velocity;
  out << std::scientific << std::setprecision(3)
      << " ch_solver=" << diag.ch_solver_name
      << " ch_res=" << diag.ch_inner_residual
      << " ch_eq_res=" << diag.ch_equation_residual
      << " dt_lim=" << diag.dt_limit_active
      << " dt_ratio=" << diag.dt_limit_ratio
      << " dt_adv=" << diag.dt_limit_advective
      << " dt_cap=" << diag.dt_limit_capillary
      << " dt_ch=" << diag.dt_limit_ch_explicit
      << " mom_solver=" << diag.momentum_solver_name
      << " mom_res=" << diag.momentum_residual
      << " p_solver=" << diag.pressure_solver_name
      << " p_corr_res=" << diag.pressure_correction_residual
      << " coupling_res=" << diag.coupling_residual
      << " dt_src=" << diag.dt_limit_source;
  out << std::fixed << std::setprecision(6)
      << " bc_pre=" << diag.boundary_speed_pre_correction
      << " bc_post=" << diag.boundary_speed_post_correction << " free_energy=" << diag.total_free_energy
      << " rho=[" << diag.rho_min << "," << diag.rho_max << "]"
      << " eta=[" << diag.eta_min << "," << diag.eta_max << "]"
      << " ch_it=" << diag.ch_iterations
      << " outer_it=" << diag.coupling_iterations
      << " p_it=" << diag.pressure_iterations
      << " mom_it=" << diag.momentum_iterations;
  return out.str();
}

void Solver::print_diagnostics(int step, const Diagnostics &diag) const {
  std::cout << format_step_report(step, static_cast<double>(step) * cfg_.dt, diag) << "\n";
}


} // namespace ding
