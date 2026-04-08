#include "core/base/config.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <stdexcept>

namespace ding {

namespace {

std::string trim(const std::string &text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1);
}

bool parse_bool(const std::string &value) {
  std::string lower = value;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return lower == "1" || lower == "true" || lower == "yes" || lower == "on";
}

BoundaryConditionType parse_boundary_condition_type(const std::string &value) {
  std::string lower = value;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (lower.empty() || lower == "unset" || lower == "default") {
    return BoundaryConditionType::unset;
  }
  if (lower == "dirichlet") {
    return BoundaryConditionType::dirichlet;
  }
  if (lower == "neumann") {
    return BoundaryConditionType::neumann;
  }
  throw std::runtime_error("unsupported boundary condition type: " + value);
}

} // namespace

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
  auto set_bc = [&](const char *type_key, const char *value_key, BoundaryConditionSpec &bc) {
    if (kv.count(type_key) != 0) {
      bc.type = parse_boundary_condition_type(kv[type_key]);
    }
    if (kv.count(value_key) != 0) {
      bc.value = std::stod(kv[value_key]);
    }
  };

  set_string("name", cfg.name);
  set_string("mode", cfg.mode);
  set_string("ch_solver", cfg.ch_solver);
  set_string("phase_initializer", cfg.phase_initializer);
  set_string("velocity_profile", cfg.velocity_profile);
  set_string("momentum_advection_scheme", cfg.momentum_advection_scheme);
  set_string("pressure_scheme", cfg.pressure_scheme);
  set_string("output_dir", cfg.output_dir);
  set_string("restart_file", cfg.restart_file);
  set_string("petsc_python_executable", cfg.petsc_python_executable);
  set_string("petsc_solver_script", cfg.petsc_solver_script);
  set_string("petsc_solver_config", cfg.petsc_solver_config);
  set_string("hydea_solver_script", cfg.hydea_solver_script);
  set_string("hydea_solver_config", cfg.hydea_solver_config);
  set_string("hydea_model_path", cfg.hydea_model_path);
  set_string("analysis_mode", cfg.analysis_mode);
  set_string("analysis_case_group", cfg.analysis_case_group);
  set_string("analysis_initial_guess", cfg.analysis_initial_guess);
  set_string("analysis_nullspace_treatment", cfg.analysis_nullspace_treatment);
  set_string("analysis_spectrum_iterations", cfg.analysis_spectrum_iterations);
  set_int("petsc_ch_log_every", cfg.petsc_ch_log_every);
  set_int("petsc_pressure_log_every", cfg.petsc_pressure_log_every);
  set_int("analysis_trigger_step", cfg.analysis_trigger_step);
  set_int("nx", cfg.nx);
  set_int("ny", cfg.ny);
  set_int("ghost", cfg.ghost);
  set_int("steps", cfg.steps);
  set_int("output_every", cfg.output_every);
  set_int("write_every", cfg.write_every);
  set_int("restart_every", cfg.restart_every);
  set_int("coupling_iterations", cfg.coupling_iterations);
  set_int("ch_inner_iterations", cfg.ch_inner_iterations);
  set_int("momentum_iterations", cfg.momentum_iterations);
  set_int("poisson_iterations", cfg.poisson_iterations);
  set_int("surface_tension_smoothing_passes", cfg.surface_tension_smoothing_passes);
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
  set_double("surface_tension_multiplier", cfg.surface_tension_multiplier);
  set_double("surface_tension_smoothing_weight", cfg.surface_tension_smoothing_weight);
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
  set_double("zalesak_slot_width", cfg.zalesak_slot_width);
  set_double("zalesak_slot_depth", cfg.zalesak_slot_depth);
  set_double("advect_u", cfg.advect_u);
  set_double("advect_v", cfg.advect_v);
  set_double("rotation_center_x", cfg.rotation_center_x);
  set_double("rotation_center_y", cfg.rotation_center_y);
  set_double("angular_velocity", cfg.angular_velocity);
  set_double("top_wall_velocity_x", cfg.top_wall_velocity_x);
  set_double("bottom_wall_velocity_x", cfg.bottom_wall_velocity_x);
  set_double("analysis_interface_band_multiplier", cfg.analysis_interface_band_multiplier);
  set_double("check_mass_drift_max", cfg.check_mass_drift_max);
  set_double("check_divergence_max", cfg.check_divergence_max);
  set_double("check_mu_max", cfg.check_mu_max);
  set_double("check_velocity_max", cfg.check_velocity_max);
  set_bc("pressure_bc_left_type", "pressure_bc_left_value", cfg.pressure_bc_left);
  set_bc("pressure_bc_right_type", "pressure_bc_right_value", cfg.pressure_bc_right);
  set_bc("pressure_bc_bottom_type", "pressure_bc_bottom_value", cfg.pressure_bc_bottom);
  set_bc("pressure_bc_top_type", "pressure_bc_top_value", cfg.pressure_bc_top);
  set_bc("u_bc_left_type", "u_bc_left_value", cfg.u_bc_left);
  set_bc("u_bc_right_type", "u_bc_right_value", cfg.u_bc_right);
  set_bc("u_bc_bottom_type", "u_bc_bottom_value", cfg.u_bc_bottom);
  set_bc("u_bc_top_type", "u_bc_top_value", cfg.u_bc_top);
  set_bc("v_bc_left_type", "v_bc_left_value", cfg.v_bc_left);
  set_bc("v_bc_right_type", "v_bc_right_value", cfg.v_bc_right);
  set_bc("v_bc_bottom_type", "v_bc_bottom_value", cfg.v_bc_bottom);
  set_bc("v_bc_top_type", "v_bc_top_value", cfg.v_bc_top);
  set_bool("periodic_x", cfg.periodic_x);
  set_bool("periodic_y", cfg.periodic_y);
  set_bool("verbose", cfg.verbose);
  set_bool("write_vtk", cfg.write_vtk);
  set_bool("restart", cfg.restart);
  set_bool("write_restart", cfg.write_restart);
  set_bool("invert_phase", cfg.invert_phase);
  set_bool("use_phase_clamp_debug_only", cfg.use_phase_clamp_debug_only);
  set_bool("analysis_enabled", cfg.analysis_enabled);
  return cfg;
}

} // namespace ding
