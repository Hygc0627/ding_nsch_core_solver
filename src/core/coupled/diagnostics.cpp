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
      << " mom_solver=" << diag.momentum_solver_name
      << " mom_res=" << diag.momentum_residual
      << " p_solver=" << diag.pressure_solver_name
      << " p_corr_res=" << diag.pressure_correction_residual
      << " coupling_res=" << diag.coupling_residual;
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
