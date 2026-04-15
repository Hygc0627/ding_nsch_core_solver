#include "solver.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace ding {

namespace {

double signed_distance_box(double x, double y, double cx, double cy, double hx, double hy) {
  const double qx = std::abs(x - cx) - hx;
  const double qy = std::abs(y - cy) - hy;
  const double ox = std::max(qx, 0.0);
  const double oy = std::max(qy, 0.0);
  return std::hypot(ox, oy) + std::min(std::max(qx, qy), 0.0);
}

} // namespace

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

bool Solver::is_advection_only_mode() const { return cfg_.mode == "advection_only"; }

bool Solver::is_single_phase_mode() const { return cfg_.mode == "single_phase"; }

bool Solver::is_phase_transport_frozen() const { return is_single_phase_mode() || cfg_.freeze_ch; }

bool Solver::should_solve_cahn_hilliard() const { return !is_advection_only_mode() && !is_phase_transport_frozen(); }

bool Solver::should_compute_chemical_potential() const { return !is_advection_only_mode() && !is_single_phase_mode(); }

void Solver::initialize() {
  validate_boundary_configuration();
  if (cfg_.surface_tension_smoothing_passes < 0) {
    throw std::runtime_error("surface_tension_smoothing_passes must be non-negative");
  }
  if (cfg_.surface_tension_smoothing_weight < 0.0 || cfg_.surface_tension_smoothing_weight > 0.25) {
    throw std::runtime_error("surface_tension_smoothing_weight must lie in [0, 0.25]");
  }
  restarted_from_snapshot_ = false;
  restart_step_ = 0;
  last_ch_iterations_ = 0;
  last_pressure_iterations_ = 0;
  last_momentum_iterations_ = 0;
  last_coupling_iterations_ = 0;
  last_ch_solver_name_ = "NotSolved";
  last_momentum_solver_name_ = "NotSolved";
  last_pressure_solver_name_ = "NotSolved";
  if (!should_solve_cahn_hilliard()) {
    last_ch_solver_name_ = "NotUsed";
  }
  if (cfg_.restart || !cfg_.restart_file.empty()) {
    load_restart_snapshot();
    return;
  }

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
  refresh_chemical_potential_for_current_phase();
  clear_surface_tension_force();
  initial_mass_ = compute_mass(c_);
  pressure_.fill(0.0);
  apply_pressure_bc(pressure_);
  pressure_previous_step_ = pressure_;
  phase_explicit_operator_prev_.fill(0.0);
}

void Solver::initialize_phase() {
  if (cfg_.phase_initializer == "zalesak_disk") {
    initialize_zalesak_disk();
    return;
  }

  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      const double x = (static_cast<double>(i) + 0.5) * dx_;
      const double y = (static_cast<double>(j) + 0.5) * dy_;
      double value = 1.0;

      if (is_single_phase_mode()) {
        c_(i, j) = cfg_.invert_phase ? 0.0 : 1.0;
        continue;
      }

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

      c_(i, j) = cfg_.invert_phase ? (1.0 - value) : value;
    }
  }
}

void Solver::initialize_zalesak_disk() {
  const double cx = cfg_.interface_center_x;
  const double cy = cfg_.interface_center_y;
  const double radius = cfg_.interface_radius;
  const double slot_half_width = 0.5 * cfg_.zalesak_slot_width;
  const double slot_top = cy + radius;
  const double slot_bottom = slot_top - cfg_.zalesak_slot_depth;
  const double slot_cy = 0.5 * (slot_top + slot_bottom);
  const double slot_half_height = 0.5 * (slot_top - slot_bottom);
  const double smoothing = std::max(2.0 * std::sqrt(2.0) * cfg_.cn, 1.0e-12);

  for (int i = 0; i < cfg_.nx; ++i) {
    for (int j = 0; j < cfg_.ny; ++j) {
      const double x = (static_cast<double>(i) + 0.5) * dx_;
      const double y = (static_cast<double>(j) + 0.5) * dy_;
      const double disk_sd = std::hypot(x - cx, y - cy) - radius;
      const double slot_sd = signed_distance_box(x, y, cx, slot_cy, slot_half_width, slot_half_height);
      const double shape_sd = std::max(disk_sd, -slot_sd);
      const double value = 0.5 - 0.5 * std::tanh(shape_sd / smoothing);
      c_(i, j) = cfg_.invert_phase ? (1.0 - value) : value;
    }
  }
}

void Solver::initialize_velocity() {
  if (cfg_.velocity_profile == "solid_body_rotation") {
    fill_solid_body_rotation_velocity(u_, v_);
    return;
  }

  const BoundaryConditionSpec bottom_u_bc = effective_u_bc(BoundarySide::bottom);
  const BoundaryConditionSpec top_u_bc = effective_u_bc(BoundarySide::top);
  const bool impose_couette_profile =
      !cfg_.periodic_y &&
      bottom_u_bc.type == BoundaryConditionType::dirichlet &&
      top_u_bc.type == BoundaryConditionType::dirichlet &&
      (std::abs(top_u_bc.value) > 0.0 || std::abs(bottom_u_bc.value) > 0.0);
  for (int i = 0; i < u_.nx; ++i) {
    for (int j = 0; j < u_.ny; ++j) {
      if (impose_couette_profile) {
        const double y = (static_cast<double>(j) + 0.5) * dy_;
        u_(i, j) = bottom_u_bc.value + (top_u_bc.value - bottom_u_bc.value) * y / std::max(cfg_.ly, 1.0e-30);
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

void Solver::fill_solid_body_rotation_velocity(Field2D &u_state, Field2D &v_state) const {
  for (int i = -u_state.ghost; i < u_state.nx + u_state.ghost; ++i) {
    for (int j = -u_state.ghost; j < u_state.ny + u_state.ghost; ++j) {
      const double y = (static_cast<double>(j) + 0.5) * dy_;
      u_state(i, j) = -cfg_.angular_velocity * (y - cfg_.rotation_center_y);
    }
  }
  for (int i = -v_state.ghost; i < v_state.nx + v_state.ghost; ++i) {
    for (int j = -v_state.ghost; j < v_state.ny + v_state.ghost; ++j) {
      const double x = (static_cast<double>(i) + 0.5) * dx_;
      v_state(i, j) = cfg_.angular_velocity * (x - cfg_.rotation_center_x);
    }
  }
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

  if (is_advection_only_mode()) {
    ch_residual = solve_phase_advection_only(u_, v_);
    update_materials();
    mu_.fill(0.0);
    apply_scalar_bc(mu_);
    last_ch_iterations_ = 3;
    last_ch_solver_name_ = "PureAdvectionSSPRK3[WENO5]";
    last_momentum_iterations_ = 0;
    last_pressure_iterations_ = 0;
    last_coupling_iterations_ = 0;
    last_momentum_solver_name_ = "NotUsed";
    last_pressure_solver_name_ = "NotUsed";
    last_boundary_speed_pre_correction_ = 0.0;
    last_boundary_speed_post_correction_ = 0.0;
    last_ch_equation_residual_ = 0.0;
    last_diag_ = compute_diagnostics();
    last_diag_.ch_inner_residual = ch_residual;
    last_diag_.ch_equation_residual = 0.0;
    last_diag_.coupling_residual = 0.0;
    last_diag_.pressure_correction_residual = 0.0;
    last_diag_.momentum_residual = 0.0;
    return true;
  }

  current_coupling_iteration_ = 0;
  const PhaseStepReport phase_report = advance_phase_state(step);
  ch_residual = phase_report.ch_residual;

  Field2D u_adv(u_.nx, u_.ny, u_.ghost, 0.0);
  Field2D v_adv(v_.nx, v_.ny, v_.ghost, 0.0);
  compute_momentum_fluxes(u_adv, v_adv);
  momentum_residual = solve_momentum_predictor(u_adv, v_adv, step);
  maybe_run_pressure_analysis_frozen();
  pressure_residual = solve_pressure_correction();
  apply_pressure_velocity_correction();

  refresh_chemical_potential_for_current_phase();
  last_coupling_iterations_ = phase_report.outer_iterations;
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
    if (restarted_from_snapshot_) {
      load_history_csv();
    }
    open_history_csv_stream();
    const int start_step = restarted_from_snapshot_ ? restart_step_ : 0;
    const double start_time = static_cast<double>(start_step) * cfg_.dt;
    while (!history_.empty() && history_.back().step > start_step) {
      history_.pop_back();
    }
    const bool need_start_entry = history_.empty() || history_.back().step != start_step;
    if (need_start_entry) {
      history_.push_back({start_step, start_time, last_diag_});
      append_history_csv_entry(start_step, start_time, last_diag_);
    }

    if (restarted_from_snapshot_) {
      std::ostringstream resume;
      resume << std::setprecision(17)
             << "RESTART file=" << (cfg_.restart_file.empty() ? restart_snapshot_path() : cfg_.restart_file)
             << " step=" << restart_step_ << " time=" << start_time;
      log_message(resume.str());
      if (cfg_.print_step_log) {
        std::cout << resume.str() << "\n";
      }
    } else {
      const std::string step_report = format_step_report(0, 0.0, last_diag_);
      log_message(step_report);
      if (cfg_.verbose || cfg_.print_step_log) {
        std::cout << step_report << "\n";
      }
      if (cfg_.write_vtk) {
        write_visualization(0);
      }
      if (should_write_restart(0)) {
        write_restart_snapshot(0);
      }
    }

    for (int step = start_step + 1; step <= cfg_.steps; ++step) {
      if (!advance_one_timestep(step - 1)) {
        log_message("ABORT step=" + std::to_string(step) + " reason=advance_one_timestep returned false");
        close_history_csv_stream();
        close_case_log();
        stop_petsc_pressure_worker();
        stop_dcdm_direction_worker();
        return false;
      }

      const double time = static_cast<double>(step) * cfg_.dt;
      if (should_write_restart(step)) {
        write_restart_snapshot(step);
      }
      const std::string step_report = format_step_report(step, time, last_diag_);
      log_message(step_report);
      if ((cfg_.verbose || cfg_.print_step_log) && (step % cfg_.output_every == 0 || step == cfg_.steps)) {
        std::cout << step_report << "\n";
      }
      if (cfg_.write_vtk && (step % cfg_.write_every == 0 || step == cfg_.steps)) {
        write_visualization(step);
      }
      history_.push_back({step, time, last_diag_});
      append_history_csv_entry(step, time, last_diag_);

      if (!std::isfinite(last_diag_.mass_drift) || !std::isfinite(last_diag_.divergence_l2) ||
          !std::isfinite(last_diag_.max_velocity) || !std::isfinite(last_diag_.max_abs_mu)) {
        log_message("ABORT step=" + std::to_string(step) + " reason=non-finite diagnostic detected");
        std::cerr << "non-finite diagnostic detected\n";
        close_history_csv_stream();
        close_case_log();
        stop_petsc_pressure_worker();
        stop_dcdm_direction_worker();
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
    close_history_csv_stream();
    close_case_log();
    stop_petsc_pressure_worker();
    stop_dcdm_direction_worker();
    std::cout << summary.str() << "\n";
    return ok;
  } catch (const std::exception &ex) {
    log_message("ERROR step=" + std::to_string(current_step_index_ + 1) + " message=" + ex.what());
    close_history_csv_stream();
    close_case_log();
    stop_petsc_pressure_worker();
    stop_dcdm_direction_worker();
    throw;
  }
}

} // namespace ding
