#pragma once

#include "core/linear_algebra/ch_sparse_krylov.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace ding {

struct Config {
  std::string name = "unnamed";
  std::string mode = "coupled";
  std::string momentum_advection_scheme = "centered";
  std::string pressure_scheme = "jacobi";
  std::string output_dir = "output";
  std::string petsc_python_executable = "python3";
  std::string petsc_solver_script = "python/petsc_pressure_solver.py";
  std::string petsc_solver_config = "python/petsc_pressure_options.py";
  std::string hydea_solver_script = "python/hydea_pressure_solver.py";
  std::string hydea_solver_config = "python/hydea_pressure_options.py";
  std::string hydea_model_path = "";
  int petsc_pressure_log_every = 50;
  int nx = 64;
  int ny = 64;
  int ghost = 3;
  int steps = 1;
  int output_every = 10;
  int write_every = 10;
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
  double stabilization_a1 = 4.0;
  double stabilization_a2 = 0.5;
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
  double advect_u = 0.0;
  double advect_v = 0.0;
  double check_mass_drift_max = 1.0e-6;
  double check_divergence_max = 1.0e-4;
  double check_mu_max = 1.0e4;
  double check_velocity_max = 10.0;
  bool periodic_x = true;
  bool periodic_y = true;
  bool verbose = true;
  bool write_vtk = true;
  bool use_phase_clamp_debug_only = false;
};

struct Field2D {
  int nx = 0;
  int ny = 0;
  int ghost = 0;
  std::vector<double> data;

  Field2D() = default;
  Field2D(int nx_, int ny_, int ghost_, double value = 0.0);

  double &operator()(int i, int j);
  double operator()(int i, int j) const;
  void fill(double value);
};

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
  int coupling_iterations = 0;
  int pressure_iterations = 0;
  int momentum_iterations = 0;
};

struct HistoryEntry {
  int step = 0;
  double time = 0.0;
  Diagnostics diag{};
};

class Solver {
public:
  explicit Solver(Config cfg);
  void initialize();
  bool run();

private:
  Config cfg_;
  double dx_;
  double dy_;

  Field2D c_;
  Field2D c_previous_step_;
  Field2D c_two_steps_back_;
  Field2D mu_;
  Field2D rho_;
  Field2D eta_;
  Field2D rho_previous_step_;
  Field2D eta_previous_step_;
  Field2D rho_mid_;
  Field2D eta_mid_;
  Field2D pressure_;
  Field2D pressure_previous_step_;
  Field2D pressure_correction_;
  Field2D u_;
  Field2D v_;
  Field2D u_previous_step_;
  Field2D v_previous_step_;
  Field2D u_star_;
  Field2D v_star_;
  Field2D surface_fx_cell_;
  Field2D surface_fy_cell_;
  Field2D surface_fx_u_;
  Field2D surface_fy_v_;
  Field2D phase_advection_rhs_;
  Field2D phase_advection_rhs_prev_;
  Field2D phase_explicit_operator_prev_;
  Field2D momentum_u_rhs_prev_;
  Field2D momentum_v_rhs_prev_;

  double initial_mass_ = 0.0;
  Diagnostics last_diag_{};
  std::vector<HistoryEntry> history_;
  double last_boundary_speed_pre_correction_ = 0.0;
  double last_boundary_speed_post_correction_ = 0.0;
  double last_ch_equation_residual_ = 0.0;
  int last_pressure_iterations_ = 0;
  int last_momentum_iterations_ = 0;
  int last_coupling_iterations_ = 0;
  int current_step_index_ = 0;
  int current_coupling_iteration_ = 0;
  mutable bool ch_operator_matrices_ready_ = false;
  mutable ch_sparse_krylov::SparseMatrixCSR ch_laplacian_matrix_;
  mutable ch_sparse_krylov::SparseMatrixCSR ch_biharmonic_matrix_;
  mutable double ch_cached_alpha0_ = -1.0;
  mutable ch_sparse_krylov::SparseMatrixCSR ch_linear_system_matrix_;
  mutable ch_sparse_krylov::KrylovPreconditioner ch_linear_system_preconditioner_;

  void initialize_phase();
  void initialize_velocity();
  void ensure_ch_operator_matrices() const;
  void ensure_ch_linear_system(double alpha0) const;

  void apply_scalar_bc(Field2D &field) const;
  void apply_u_bc(Field2D &field) const;
  void apply_v_bc(Field2D &field) const;

  void update_materials();
  void update_materials_from_phase(const Field2D &c_state, Field2D &rho_state, Field2D &eta_state) const;
  void update_midpoint_materials();
  void update_chemical_potential(const Field2D &c_state, Field2D &mu_state) const;
  void update_surface_tension_force(const Field2D &c_old, const Field2D &c_new);

  bool advance_one_timestep(int step);
  double solve_cahn_hilliard_semi_implicit(const Field2D &u_adv, const Field2D &v_adv, int step);
  void build_phase_advection_fluxes(const Field2D &c_state, const Field2D &u_adv, const Field2D &v_adv,
                                    Field2D &adv_flux_x, Field2D &adv_flux_y, Field2D &adv_rhs) const;
  void build_phase_diffusion_fluxes(const Field2D &scalar_field, const Field2D &mobility, Field2D &flux_x,
                                    Field2D &flux_y, Field2D &divergence) const;
  void build_phase_explicit_operator(const Field2D &c_state, const Field2D &u_adv, const Field2D &v_adv,
                                     Field2D &explicit_operator) const;
  void solve_phase_linear_system_eq25(const Field2D &rhs_field, double target_mean, Field2D &c_state,
                                      double &iterate_residual, double &equation_residual) const;
  void compute_momentum_fluxes(Field2D &u_adv, Field2D &v_adv) const;
  double solve_momentum_predictor(const Field2D &u_adv, const Field2D &v_adv, int step);
  double solve_pressure_correction();
  void apply_pressure_velocity_correction();
  double compute_coupling_residuals(const Field2D &c_old, const Field2D &u_old, const Field2D &v_old,
                                    const Field2D &p_old) const;

  double compute_mass(const Field2D &field) const;
  double compute_free_energy() const;
  Diagnostics compute_diagnostics() const;
  void print_diagnostics(int step, const Diagnostics &diag) const;
  void write_visualization(int step) const;
  void write_pvd_index() const;
  void write_summary_csv() const;
  void write_history_csv() const;
  void write_final_cell_fields_csv() const;
  std::string case_output_dir() const;
  std::string pressure_solver_dir() const;

  double laplacian_center(const Field2D &field, int i, int j) const;
  double grad_center_x(const Field2D &field, int i, int j) const;
  double grad_center_y(const Field2D &field, int i, int j) const;
  double grad_face_x_centered(const Field2D &field, int i_face, int j) const;
  double grad_face_y_centered(const Field2D &field, int i, int j_face) const;
  double cell_centered_u(const Field2D &u_state, int i, int j) const;
  double cell_centered_v(const Field2D &v_state, int i, int j) const;
  double rho_u_face(int i, int j) const;
  double rho_v_face(int i, int j) const;
  double eta_u_face(int i, int j) const;
  double eta_v_face(int i, int j) const;
  double rho_corner(int i, int j) const;
  double eta_corner(int i, int j) const;
  double rho_u_face(const Field2D &rho_field, int i, int j) const;
  double rho_v_face(const Field2D &rho_field, int i, int j) const;
  double eta_u_face(const Field2D &eta_field, int i, int j) const;
  double eta_v_face(const Field2D &eta_field, int i, int j) const;
  double rho_corner(const Field2D &rho_field, int i, int j) const;
  double eta_corner(const Field2D &eta_field, int i, int j) const;
  double divergence_cell(const Field2D &u_state, const Field2D &v_state, int i, int j) const;
  double pressure_gradient_u_face(const Field2D &pressure_like, int i, int j) const;
  double pressure_gradient_v_face(const Field2D &pressure_like, int i, int j) const;
  double stress_divergence_u(const Field2D &u_state, const Field2D &v_state, int i, int j) const;
  double stress_divergence_v(const Field2D &u_state, const Field2D &v_state, int i, int j) const;
  double stress_divergence_u(const Field2D &u_state, const Field2D &v_state, const Field2D &eta_field, int i,
                             int j) const;
  double stress_divergence_v(const Field2D &u_state, const Field2D &v_state, const Field2D &eta_field, int i,
                             int j) const;
  double momentum_u_face_value(const Field2D &u_state, int i, int j) const;
  double momentum_v_face_value(const Field2D &v_state, int i, int j) const;
  double reconstruct_with_scheme(double vel, double q_mm, double q_m, double q_p, double q_pp) const;
  double phase_weno_x_face_value(const Field2D &c_state, const Field2D &u_adv, int i, int j) const;
  double phase_weno_y_face_value(const Field2D &c_state, const Field2D &v_adv, int i, int j) const;
  double weno5_left(double v1, double v2, double v3, double v4, double v5) const;
  double weno5_right(double v1, double v2, double v3, double v4, double v5) const;
  double second_order_upwind(double vel, double q_mm, double q_m, double q_p, double q_pp) const;
  double clamp_phase_debug(double value) const;
  double field_diff_l2(const Field2D &a, const Field2D &b, int i_begin, int i_end, int j_begin, int j_end) const;
  void subtract_mean(Field2D &field) const;
  double periodic_boundary_speed_stat(const Field2D &u_state, const Field2D &v_state) const;
  double solve_pressure_correction_jacobi();
  double solve_pressure_correction_icpcg();
  double solve_pressure_correction_liu_split_icpcg();
  double solve_pressure_correction_petsc();
  double solve_pressure_correction_hydea();
  bool use_liu_pressure_split() const;
  double liu_split_reference_density() const;
  void build_liu_split_pressure_extrapolation(Field2D &pressure_extrapolated) const;
  double liu_split_explicit_divergence(const Field2D &pressure_extrapolated, int i, int j) const;
};

Config load_config(const std::string &path);

} // namespace ding
