#include "solver.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace ding {

Solver::Solver(Config cfg)
    : cfg_(std::move(cfg)),
      grid_(cfg_),
      dx_(grid_.dx),
      dy_(grid_.dy),
      c_(grid_.nx, grid_.ny, grid_.ghost, 0.0),
      c_previous_step_(grid_.nx, grid_.ny, grid_.ghost, 0.0),
      c_two_steps_back_(grid_.nx, grid_.ny, grid_.ghost, 0.0),
      mu_(grid_.nx, grid_.ny, grid_.ghost, 0.0),
      rho_(grid_.nx, grid_.ny, grid_.ghost, 1.0),
      eta_(grid_.nx, grid_.ny, grid_.ghost, 1.0),
      rho_previous_step_(grid_.nx, grid_.ny, grid_.ghost, 1.0),
      eta_previous_step_(grid_.nx, grid_.ny, grid_.ghost, 1.0),
      rho_mid_(grid_.nx, grid_.ny, grid_.ghost, 1.0),
      eta_mid_(grid_.nx, grid_.ny, grid_.ghost, 1.0),
      pressure_(grid_.nx, grid_.ny, grid_.ghost, 0.0),
      pressure_previous_step_(grid_.nx, grid_.ny, grid_.ghost, 0.0),
      pressure_correction_(grid_.nx, grid_.ny, grid_.ghost, 0.0),
      u_(grid_.nx + 1, grid_.ny, grid_.ghost, 0.0),
      v_(grid_.nx, grid_.ny + 1, grid_.ghost, 0.0),
      u_previous_step_(grid_.nx + 1, grid_.ny, grid_.ghost, 0.0),
      v_previous_step_(grid_.nx, grid_.ny + 1, grid_.ghost, 0.0),
      u_star_(grid_.nx + 1, grid_.ny, grid_.ghost, 0.0),
      v_star_(grid_.nx, grid_.ny + 1, grid_.ghost, 0.0),
      surface_fx_cell_(grid_.nx, grid_.ny, grid_.ghost, 0.0),
      surface_fy_cell_(grid_.nx, grid_.ny, grid_.ghost, 0.0),
      surface_fx_u_(grid_.nx + 1, grid_.ny, grid_.ghost, 0.0),
      surface_fy_v_(grid_.nx, grid_.ny + 1, grid_.ghost, 0.0),
      phase_advection_rhs_(grid_.nx, grid_.ny, grid_.ghost, 0.0),
      phase_advection_rhs_prev_(grid_.nx, grid_.ny, grid_.ghost, 0.0),
      phase_explicit_operator_prev_(grid_.nx, grid_.ny, grid_.ghost, 0.0),
      momentum_u_rhs_prev_(grid_.nx + 1, grid_.ny, grid_.ghost, 0.0),
      momentum_v_rhs_prev_(grid_.nx, grid_.ny + 1, grid_.ghost, 0.0) {}

void Solver::initialize() {
  last_ch_iterations_ = 0;
  last_pressure_iterations_ = 0;
  last_momentum_iterations_ = 0;
  last_coupling_iterations_ = 0;
  last_ch_solver_name_ = "NotSolved";
  last_momentum_solver_name_ = "NotSolved";
  last_pressure_solver_name_ = "NotSolved";
  initialize_phase();
  initialize_velocity();
  apply_scalar_bc(c_);
  c_previous_step_ = c_;
  c_two_steps_back_ = c_;
  u_previous_step_ = u_;
  v_previous_step_ = v_;
  update_materials();
  rho_previous_step_ = rho_;
  eta_previous_step_ = eta_;
  rho_mid_ = rho_;
  eta_mid_ = eta_;
  update_chemical_potential(c_, mu_);
  initial_mass_ = compute_mass(c_);
  pressure_.fill(0.0);
  apply_scalar_bc(pressure_);
  pressure_previous_step_ = pressure_;
  phase_explicit_operator_prev_.fill(0.0);
}

void Solver::initialize_phase() {
  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      const double x = (static_cast<double>(i) + 0.5) * dx_;
      const double y = (static_cast<double>(j) + 0.5) * dy_;
      double value = 1.0;

      if (cfg_.mode == "coupled" || cfg_.mode == "equilibrium" || cfg_.mode == "ch_only") {
        if (cfg_.interface_radius > 0.0) {
          const double rx = x - cfg_.interface_center_x;
          const double ry = y - cfg_.interface_center_y;
          const double r = std::sqrt(rx * rx + ry * ry);
          // For W(c)=0.25*c^2*(1-c)^2, the equilibrium tanh profile uses 2*sqrt(2)*Cn.
          value = 0.5 - 0.5 * std::tanh((r - cfg_.interface_radius) / (2.0 * std::sqrt(2.0) * cfg_.cn));
        } else {
          const double y0 = cfg_.interface_center_y +
                            cfg_.interface_amplitude *
                                std::cos(2.0 * M_PI * cfg_.interface_wavenumber * x / std::max(cfg_.lx, 1.0e-12));
          value = 0.5 - 0.5 * std::tanh((y - y0) / (2.0 * std::sqrt(2.0) * cfg_.cn));
        }
      }

      c_(i, j) = value;
    }
  }
}

void Solver::initialize_velocity() {
  const bool impose_couette_profile =
      !cfg_.periodic_y &&
      (std::abs(cfg_.top_wall_velocity_x) > 0.0 || std::abs(cfg_.bottom_wall_velocity_x) > 0.0);
  for (int i = 0; i < u_.nx; ++i) {
    for (int j = 0; j < u_.ny; ++j) {
      if (impose_couette_profile) {
        const double y = (static_cast<double>(j) + 0.5) * dy_;
        u_(i, j) = cfg_.bottom_wall_velocity_x +
                   (cfg_.top_wall_velocity_x - cfg_.bottom_wall_velocity_x) * y / std::max(cfg_.ly, 1.0e-30);
      } else {
        u_(i, j) = cfg_.advect_u;
      }
    }
  }
  for (int i = 0; i < v_.nx; ++i) {
    for (int j = 0; j < v_.ny; ++j) {
      v_(i, j) = cfg_.advect_v;
    }
  }
  apply_u_velocity_bc(u_);
  apply_v_bc(v_);
}


bool Solver::advance_one_timestep(int step) {
  current_step_index_ = step;
  c_two_steps_back_ = c_previous_step_;
  c_previous_step_ = c_;
  rho_previous_step_ = rho_;
  eta_previous_step_ = eta_;
  u_previous_step_ = u_;
  v_previous_step_ = v_;

  double ch_residual = 0.0;
  double coupling_residual = 0.0;
  double pressure_residual = 0.0;
  double momentum_residual = 0.0;
  const int outer_iterations = 1;
  current_coupling_iteration_ = 0;

  // Ding 2007 §3:
  // (1) solve CH with u^n
  // (2) compute surface tension with 0.5*(C^n + C^{n+1})
  // (3) solve NS/projection once to obtain u^{n+1}
  ch_residual = solve_cahn_hilliard_semi_implicit(u_, v_, step);
  update_materials();
  update_midpoint_materials();
  update_surface_tension_force(c_previous_step_, c_);

  Field2D u_adv(u_.nx, u_.ny, u_.ghost, 0.0);
  Field2D v_adv(v_.nx, v_.ny, v_.ghost, 0.0);
  compute_momentum_fluxes(u_adv, v_adv);
  momentum_residual = solve_momentum_predictor(u_adv, v_adv, step);
  pressure_residual = solve_pressure_correction();
  apply_pressure_velocity_correction();

  update_chemical_potential(c_, mu_);
  last_coupling_iterations_ = outer_iterations;
  last_diag_ = compute_diagnostics();
  last_diag_.ch_inner_residual = ch_residual;
  last_diag_.ch_equation_residual = last_ch_equation_residual_;
  last_diag_.coupling_residual = coupling_residual;
  last_diag_.pressure_correction_residual = pressure_residual;
  last_diag_.momentum_residual = momentum_residual;
  return true;
}


bool Solver::run() {
  open_case_log();
  try {
    log_run_header();
    initialize();
    last_diag_ = compute_diagnostics();
    history_.clear();
    history_.push_back({0, 0.0, last_diag_});
    log_message(format_step_report(0, 0.0, last_diag_));
    if (cfg_.verbose) {
      print_diagnostics(0, last_diag_);
    }
    if (cfg_.write_vtk) {
      write_visualization(0);
    }

    for (int step = 1; step <= cfg_.steps; ++step) {
      if (!advance_one_timestep(step - 1)) {
        log_message("ABORT step=" + std::to_string(step) + " reason=advance_one_timestep returned false");
        close_case_log();
        return false;
      }

      const double time = static_cast<double>(step) * cfg_.dt;
      log_message(format_step_report(step, time, last_diag_));
      if (cfg_.verbose && (step % cfg_.output_every == 0 || step == cfg_.steps)) {
        print_diagnostics(step, last_diag_);
      }
      if (cfg_.write_vtk && (step % cfg_.write_every == 0 || step == cfg_.steps)) {
        write_visualization(step);
      }
      history_.push_back({step, time, last_diag_});

      if (!std::isfinite(last_diag_.mass_drift) || !std::isfinite(last_diag_.divergence_l2) ||
          !std::isfinite(last_diag_.max_velocity) || !std::isfinite(last_diag_.max_abs_mu)) {
        log_message("ABORT step=" + std::to_string(step) + " reason=non-finite diagnostic detected");
        std::cerr << "non-finite diagnostic detected\n";
        close_case_log();
        return false;
      }
    }

    const bool ok = last_diag_.mass_drift <= cfg_.check_mass_drift_max &&
                    last_diag_.divergence_l2 <= cfg_.check_divergence_max &&
                    last_diag_.max_abs_mu <= cfg_.check_mu_max &&
                    last_diag_.max_velocity <= cfg_.check_velocity_max;

    std::ostringstream summary;
    summary << std::fixed << std::setprecision(6)
            << "RESULT name=" << cfg_.name << " status=" << (ok ? "PASS" : "FAIL")
            << " mass_drift=" << last_diag_.mass_drift << " div_l2=" << last_diag_.divergence_l2
            << " max_mu=" << last_diag_.max_abs_mu << " max_u=" << last_diag_.max_velocity;
    summary << std::scientific << std::setprecision(3)
            << " ch_res=" << last_diag_.ch_inner_residual << " ch_eq_res=" << last_diag_.ch_equation_residual
            << " coupling_res=" << last_diag_.coupling_residual
            << " p_corr_res=" << last_diag_.pressure_correction_residual;
    summary << std::fixed << std::setprecision(6)
            << " bc_pre=" << last_diag_.boundary_speed_pre_correction
            << " bc_post=" << last_diag_.boundary_speed_post_correction
            << " outer_it=" << last_diag_.coupling_iterations
            << " p_it=" << last_diag_.pressure_iterations
            << " mom_it=" << last_diag_.momentum_iterations;
    write_summary_csv();
    write_history_csv();
    write_final_cell_fields_csv();
    log_message(summary.str());
    close_case_log();
    std::cout << summary.str() << "\n";
    return ok;
  } catch (const std::exception &ex) {
    log_message("ERROR step=" + std::to_string(current_step_index_ + 1) + " message=" + ex.what());
    close_case_log();
    throw;
  }
}

} // namespace ding
