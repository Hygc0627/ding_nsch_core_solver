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
  set_double("top_wall_velocity_x", cfg.top_wall_velocity_x);
  set_double("bottom_wall_velocity_x", cfg.bottom_wall_velocity_x);
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

} // namespace ding
