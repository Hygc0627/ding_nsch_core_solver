#pragma once

#include <string>

namespace ding {

enum class BoundarySide {
  left = 0,
  right = 1,
  bottom = 2,
  top = 3,
};

enum class BoundaryConditionType {
  unset = 0,
  dirichlet = 1,
  neumann = 2,
};

struct BoundaryConditionSpec {
  BoundaryConditionType type = BoundaryConditionType::unset;
  double value = 0.0;
};

struct Config {
  std::string name = "unnamed";
  std::string mode = "coupled";
  std::string ch_solver = "sparse_pcg";
  std::string momentum_advection_scheme = "centered";
  std::string pressure_scheme = "jacobi";
  std::string phase_initializer = "default";
  std::string velocity_profile = "default";
  std::string output_dir = "output";
  std::string restart_file = "";
  std::string petsc_python_executable = "python3";
  std::string petsc_solver_script = "python/petsc_pressure_solver.py";
  std::string petsc_solver_config = "python/petsc_pressure_options.py";
  std::string hydea_solver_script = "python/hydea_pressure_solver.py";
  std::string hydea_solver_config = "python/hydea_pressure_options.py";
  std::string hydea_model_path = "";
  std::string analysis_mode = "off";
  std::string analysis_case_group = "";
  std::string analysis_initial_guess = "zero";
  std::string analysis_nullspace_treatment = "zero_mean_projection";
  std::string analysis_spectrum_iterations = "0,1,2,5,10,20,final";
  int petsc_ch_log_every = 0;
  int petsc_pressure_log_every = 50;
  int analysis_trigger_step = 1;
  int nx = 64;
  int ny = 64;
  int ghost = 3;
  int steps = 1;
  int output_every = 10;
  int write_every = 10;
  int restart_every = 0;
  int coupling_iterations = 1;
  int ch_inner_iterations = 30;
  int momentum_iterations = 50;
  int poisson_iterations = 400;
  double dt = 1.0e-4;
  double lx = 1.0;
  double ly = 1.0;
  double re = 100.0;
  double ca = 1.0;
  double pe = 100.0;
  double cn = 0.02;
  double density_ratio = 0.1;
  double viscosity_ratio = 0.1;
  double stabilization_a1 = 0.03125;
  double stabilization_a2 = 0.125;
  double surface_tension_multiplier = 1.0;
  int surface_tension_smoothing_passes = 0;
  double surface_tension_smoothing_weight = 0.0;
  double coupling_tolerance = 1.0e-6;
  double ch_tolerance = 1.0e-8;
  double pressure_tolerance = 1.0e-8;
  double momentum_tolerance = 1.0e-8;
  double body_force_x = 0.0;
  double body_force_y = 0.0;
  double interface_center_x = 0.5;
  double interface_center_y = 0.5;
  double interface_radius = 0.18;
  double interface_amplitude = 0.0;
  double interface_wavenumber = 1.0;
  double zalesak_slot_width = 0.05;
  double zalesak_slot_depth = 0.25;
  double advect_u = 0.0;
  double advect_v = 0.0;
  double rotation_center_x = 0.5;
  double rotation_center_y = 0.5;
  double angular_velocity = 0.0;
  double top_wall_velocity_x = 0.0;
  double bottom_wall_velocity_x = 0.0;
  double analysis_interface_band_multiplier = 2.0;
  double check_mass_drift_max = 1.0e-6;
  double check_divergence_max = 1.0e-4;
  double check_mu_max = 1.0e4;
  double check_velocity_max = 10.0;
  BoundaryConditionSpec pressure_bc_left{};
  BoundaryConditionSpec pressure_bc_right{};
  BoundaryConditionSpec pressure_bc_bottom{};
  BoundaryConditionSpec pressure_bc_top{};
  BoundaryConditionSpec u_bc_left{};
  BoundaryConditionSpec u_bc_right{};
  BoundaryConditionSpec u_bc_bottom{};
  BoundaryConditionSpec u_bc_top{};
  BoundaryConditionSpec v_bc_left{};
  BoundaryConditionSpec v_bc_right{};
  BoundaryConditionSpec v_bc_bottom{};
  BoundaryConditionSpec v_bc_top{};
  bool periodic_x = true;
  bool periodic_y = true;
  bool verbose = true;
  bool write_vtk = true;
  bool restart = false;
  bool write_restart = true;
  bool invert_phase = false;
  bool use_phase_clamp_debug_only = false;
  bool analysis_enabled = false;
};

Config load_config(const std::string &path);

} // namespace ding
