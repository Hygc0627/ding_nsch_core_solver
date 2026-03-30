#pragma once

#include "core/base/config.hpp"
#include "core/base/diagnostics.hpp"
#include "core/base/field.hpp"
#include "core/base/grid.hpp"
#include "core/linear_algebra/ch_sparse_krylov.hpp"

#include <fstream>
#include <string>
#include <vector>

namespace ding {

class Solver {
public:
  explicit Solver(Config cfg);
  void initialize();
  bool run();

private:
  Config cfg_;
  UniformGrid2D grid_;
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
  std::ofstream case_log_;
  double last_boundary_speed_pre_correction_ = 0.0;
  double last_boundary_speed_post_correction_ = 0.0;
  double last_ch_equation_residual_ = 0.0;
  mutable int last_ch_iterations_ = 0;
  int last_pressure_iterations_ = 0;
  int last_momentum_iterations_ = 0;
  int last_coupling_iterations_ = 0;
  int current_step_index_ = 0;
  int current_coupling_iteration_ = 0;
  mutable std::string last_ch_solver_name_ = "SparsePCG";
  std::string last_momentum_solver_name_ = "StationaryIteration";
  std::string last_pressure_solver_name_ = "Uninitialized";
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
  void apply_u_velocity_bc(Field2D &field) const;
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
  std::string format_step_report(int step, double time, const Diagnostics &diag) const;
  void write_visualization(int step) const;
  void write_pvd_index() const;
  void write_summary_csv() const;
  void write_history_csv() const;
  void write_final_cell_fields_csv() const;
  std::string case_output_dir() const;
  std::string case_log_path() const;
  std::string pressure_solver_dir() const;
  void open_case_log();
  void close_case_log();
  void log_message(const std::string &message);
  void log_run_header();

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
  double viscous_self_u(const Field2D &u_state, const Field2D &eta_field, int i, int j) const;
  double viscous_cross_u(const Field2D &v_state, const Field2D &eta_field, int i, int j) const;
  double viscous_self_v(const Field2D &v_state, const Field2D &eta_field, int i, int j) const;
  double viscous_cross_v(const Field2D &u_state, const Field2D &eta_field, int i, int j) const;
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
  double solve_pressure_correction_ildlt_pcg();
  double solve_pressure_correction_liu_split_icpcg();
  double solve_pressure_correction_liu_split_ildlt_pcg();
  double solve_pressure_correction_petsc();
  double solve_pressure_correction_hydea();
  bool use_liu_pressure_split() const;
  double liu_split_reference_density() const;
  void build_liu_split_pressure_extrapolation(Field2D &pressure_extrapolated) const;
  double liu_split_explicit_divergence(const Field2D &pressure_extrapolated, int i, int j) const;
};

} // namespace ding
