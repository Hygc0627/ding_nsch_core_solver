#include "solver.hpp"
#include "internal.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace ding {

using coupled_detail::parse_bool;
using coupled_detail::trim;

Field2D::Field2D(int nx_, int ny_, int ghost_, double value)
    : nx(nx_), ny(ny_), ghost(ghost_),
      data(static_cast<std::size_t>(nx_ + 2 * ghost_) * static_cast<std::size_t>(ny_ + 2 * ghost_), value) {}

double &Field2D::operator()(int i, int j) {
  const int ii = i + ghost;
  const int jj = j + ghost;
  return data[static_cast<std::size_t>(ii) * static_cast<std::size_t>(ny + 2 * ghost) + static_cast<std::size_t>(jj)];
}

double Field2D::operator()(int i, int j) const {
  const int ii = i + ghost;
  const int jj = j + ghost;
  return data[static_cast<std::size_t>(ii) * static_cast<std::size_t>(ny + 2 * ghost) + static_cast<std::size_t>(jj)];
}

void Field2D::fill(double value) {
  std::fill(data.begin(), data.end(), value);
}

Config load_config(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open config: " + path);
  }

  std::map<std::string, std::string> kv;
  std::string line;
  while (std::getline(in, line)) {
    const auto hash = line.find('#');
    if (hash != std::string::npos) {
      line = line.substr(0, hash);
    }
    line = trim(line);
    if (line.empty()) {
      continue;
    }
    const auto eq = line.find('=');
    if (eq == std::string::npos) {
      continue;
    }
    kv[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
  }

  Config cfg;
  auto set_int = [&](const char *key, int &value) {
    if (kv.count(key) != 0) {
      value = std::stoi(kv[key]);
    }
  };
  auto set_double = [&](const char *key, double &value) {
    if (kv.count(key) != 0) {
      value = std::stod(kv[key]);
    }
  };
  auto set_bool = [&](const char *key, bool &value) {
    if (kv.count(key) != 0) {
      value = parse_bool(kv[key]);
    }
  };
  auto set_string = [&](const char *key, std::string &value) {
    if (kv.count(key) != 0) {
      value = kv[key];
    }
  };

  set_string("name", cfg.name);
  set_string("mode", cfg.mode);
  set_string("momentum_advection_scheme", cfg.momentum_advection_scheme);
  set_string("pressure_scheme", cfg.pressure_scheme);
  set_string("output_dir", cfg.output_dir);
  set_string("petsc_python_executable", cfg.petsc_python_executable);
  set_string("petsc_solver_script", cfg.petsc_solver_script);
  set_string("petsc_solver_config", cfg.petsc_solver_config);
  set_string("hydea_solver_script", cfg.hydea_solver_script);
  set_string("hydea_solver_config", cfg.hydea_solver_config);
  set_string("hydea_model_path", cfg.hydea_model_path);
  set_int("petsc_pressure_log_every", cfg.petsc_pressure_log_every);
  set_int("nx", cfg.nx);
  set_int("ny", cfg.ny);
  set_int("ghost", cfg.ghost);
  set_int("steps", cfg.steps);
  set_int("output_every", cfg.output_every);
  set_int("write_every", cfg.write_every);
  set_int("coupling_iterations", cfg.coupling_iterations);
  set_int("ch_inner_iterations", cfg.ch_inner_iterations);
  set_int("momentum_iterations", cfg.momentum_iterations);
  set_int("poisson_iterations", cfg.poisson_iterations);
  set_double("dt", cfg.dt);
  set_double("lx", cfg.lx);
  set_double("ly", cfg.ly);
  set_double("re", cfg.re);
  set_double("ca", cfg.ca);
  set_double("pe", cfg.pe);
  set_double("cn", cfg.cn);
  set_double("density_ratio", cfg.density_ratio);
  set_double("viscosity_ratio", cfg.viscosity_ratio);
  set_double("stabilization_a1", cfg.stabilization_a1);
  set_double("stabilization_a2", cfg.stabilization_a2);
  set_double("coupling_tolerance", cfg.coupling_tolerance);
  set_double("ch_tolerance", cfg.ch_tolerance);
  set_double("pressure_tolerance", cfg.pressure_tolerance);
  set_double("momentum_tolerance", cfg.momentum_tolerance);
  set_double("body_force_x", cfg.body_force_x);
  set_double("body_force_y", cfg.body_force_y);
  set_double("interface_center_x", cfg.interface_center_x);
  set_double("interface_center_y", cfg.interface_center_y);
  set_double("interface_radius", cfg.interface_radius);
  set_double("interface_amplitude", cfg.interface_amplitude);
  set_double("interface_wavenumber", cfg.interface_wavenumber);
  set_double("advect_u", cfg.advect_u);
  set_double("advect_v", cfg.advect_v);
  set_double("check_mass_drift_max", cfg.check_mass_drift_max);
  set_double("check_divergence_max", cfg.check_divergence_max);
  set_double("check_mu_max", cfg.check_mu_max);
  set_double("check_velocity_max", cfg.check_velocity_max);
  set_bool("periodic_x", cfg.periodic_x);
  set_bool("periodic_y", cfg.periodic_y);
  set_bool("verbose", cfg.verbose);
  set_bool("write_vtk", cfg.write_vtk);
  set_bool("use_phase_clamp_debug_only", cfg.use_phase_clamp_debug_only);
  return cfg;
}

Solver::Solver(Config cfg)
    : cfg_(std::move(cfg)),
      dx_(cfg_.lx / static_cast<double>(cfg_.nx)),
      dy_(cfg_.ly / static_cast<double>(cfg_.ny)),
      c_(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0),
      c_previous_step_(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0),
      c_two_steps_back_(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0),
      mu_(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0),
      rho_(cfg_.nx, cfg_.ny, cfg_.ghost, 1.0),
      eta_(cfg_.nx, cfg_.ny, cfg_.ghost, 1.0),
      rho_previous_step_(cfg_.nx, cfg_.ny, cfg_.ghost, 1.0),
      eta_previous_step_(cfg_.nx, cfg_.ny, cfg_.ghost, 1.0),
      rho_mid_(cfg_.nx, cfg_.ny, cfg_.ghost, 1.0),
      eta_mid_(cfg_.nx, cfg_.ny, cfg_.ghost, 1.0),
      pressure_(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0),
      pressure_previous_step_(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0),
      pressure_correction_(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0),
      u_(cfg_.nx + 1, cfg_.ny, cfg_.ghost, 0.0),
      v_(cfg_.nx, cfg_.ny + 1, cfg_.ghost, 0.0),
      u_previous_step_(cfg_.nx + 1, cfg_.ny, cfg_.ghost, 0.0),
      v_previous_step_(cfg_.nx, cfg_.ny + 1, cfg_.ghost, 0.0),
      u_star_(cfg_.nx + 1, cfg_.ny, cfg_.ghost, 0.0),
      v_star_(cfg_.nx, cfg_.ny + 1, cfg_.ghost, 0.0),
      surface_fx_cell_(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0),
      surface_fy_cell_(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0),
      surface_fx_u_(cfg_.nx + 1, cfg_.ny, cfg_.ghost, 0.0),
      surface_fy_v_(cfg_.nx, cfg_.ny + 1, cfg_.ghost, 0.0),
      phase_advection_rhs_(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0),
      phase_advection_rhs_prev_(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0),
      phase_explicit_operator_prev_(cfg_.nx, cfg_.ny, cfg_.ghost, 0.0),
      momentum_u_rhs_prev_(cfg_.nx + 1, cfg_.ny, cfg_.ghost, 0.0),
      momentum_v_rhs_prev_(cfg_.nx, cfg_.ny + 1, cfg_.ghost, 0.0) {}

void Solver::initialize() {
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
  for (int i = 0; i < u_.nx; ++i) {
    for (int j = 0; j < u_.ny; ++j) {
      u_(i, j) = cfg_.advect_u;
    }
  }
  for (int i = 0; i < v_.nx; ++i) {
    for (int j = 0; j < v_.ny; ++j) {
      v_(i, j) = cfg_.advect_v;
    }
  }
  apply_u_bc(u_);
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
  initialize();
  last_diag_ = compute_diagnostics();
  history_.clear();
  history_.push_back({0, 0.0, last_diag_});
  if (cfg_.verbose) {
    print_diagnostics(0, last_diag_);
  }
  if (cfg_.write_vtk) {
    write_visualization(0);
  }

  for (int step = 1; step <= cfg_.steps; ++step) {
    if (!advance_one_timestep(step - 1)) {
      return false;
    }

    if (cfg_.verbose && (step % cfg_.output_every == 0 || step == cfg_.steps)) {
      print_diagnostics(step, last_diag_);
    }
    if (cfg_.write_vtk && (step % cfg_.write_every == 0 || step == cfg_.steps)) {
      write_visualization(step);
    }
    history_.push_back({step, static_cast<double>(step) * cfg_.dt, last_diag_});

    if (!std::isfinite(last_diag_.mass_drift) || !std::isfinite(last_diag_.divergence_l2) ||
        !std::isfinite(last_diag_.max_velocity) || !std::isfinite(last_diag_.max_abs_mu)) {
      std::cerr << "non-finite diagnostic detected\n";
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
  std::cout << summary.str() << "\n";
  return ok;
}

} // namespace ding
