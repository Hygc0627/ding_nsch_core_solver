#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#define private public
#include "core/coupled/solver.hpp"
#undef private

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

constexpr double kPi = 3.14159265358979323846;

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

ding::Config make_config() {
  ding::Config cfg;
  cfg.name = "phase4_unit";
  cfg.nx = 32;
  cfg.ny = 32;
  cfg.ghost = 3;
  cfg.dt = 1.0e-3;
  cfg.periodic_x = true;
  cfg.periodic_y = true;
  cfg.verbose = false;
  cfg.write_vtk = false;
  cfg.poisson_iterations = 5000;
  cfg.pressure_tolerance = 1.0e-10;
  cfg.momentum_iterations = 200;
  cfg.momentum_tolerance = 1.0e-10;
  cfg.density_ratio = 10.0;
  cfg.viscosity_ratio = 5.0;
  return cfg;
}

double divergence_l2(const ding::Solver &solver, const ding::Field2D &u, const ding::Field2D &v) {
  double sum = 0.0;
  for (int i = 0; i < solver.cfg_.nx; ++i) {
    for (int j = 0; j < solver.cfg_.ny; ++j) {
      const double div = solver.divergence_cell(u, v, i, j);
      sum += div * div;
    }
  }
  return std::sqrt(sum / static_cast<double>(solver.cfg_.nx * solver.cfg_.ny));
}

double field_max_abs_difference(const ding::Field2D &a, const ding::Field2D &b) {
  require(a.nx == b.nx && a.ny == b.ny && a.ghost == b.ghost, "field dimensions must match");
  double diff = 0.0;
  for (std::size_t idx = 0; idx < a.data.size(); ++idx) {
    diff = std::max(diff, std::abs(a.data[idx] - b.data[idx]));
  }
  return diff;
}

void test_midpoint_material_average() {
  ding::Solver solver(make_config());
  solver.initialize();

  for (int i = 0; i < solver.cfg_.nx; ++i) {
    for (int j = 0; j < solver.cfg_.ny; ++j) {
      solver.rho_previous_step_(i, j) = 1.0 + 0.01 * static_cast<double>(i + j);
      solver.rho_(i, j) = 2.0 + 0.02 * static_cast<double>(i - j);
      solver.eta_previous_step_(i, j) = 0.5 + 0.03 * static_cast<double>(i);
      solver.eta_(i, j) = 1.5 + 0.04 * static_cast<double>(j);
    }
  }
  solver.apply_scalar_bc(solver.rho_previous_step_);
  solver.apply_scalar_bc(solver.rho_);
  solver.apply_scalar_bc(solver.eta_previous_step_);
  solver.apply_scalar_bc(solver.eta_);
  solver.update_midpoint_materials();

  for (int i = 0; i < solver.cfg_.nx; ++i) {
    for (int j = 0; j < solver.cfg_.ny; ++j) {
      require(std::abs(solver.rho_mid_(i, j) - 0.5 * (solver.rho_previous_step_(i, j) + solver.rho_(i, j))) < 1.0e-14,
              "rho_mid must be the arithmetic average");
      require(std::abs(solver.eta_mid_(i, j) - 0.5 * (solver.eta_previous_step_(i, j) + solver.eta_(i, j))) < 1.0e-14,
              "eta_mid must be the arithmetic average");
    }
  }
}

void test_variable_coefficient_projection_reduces_divergence() {
  ding::Solver solver(make_config());
  solver.initialize();

  for (int i = 0; i < solver.cfg_.nx; ++i) {
    for (int j = 0; j < solver.cfg_.ny; ++j) {
      const double x = (static_cast<double>(i) + 0.5) * solver.dx_;
      const double y = (static_cast<double>(j) + 0.5) * solver.dy_;
      solver.c_(i, j) = 0.5 + 0.25 * std::sin(2.0 * kPi * x) * std::cos(2.0 * kPi * y);
    }
  }
  solver.apply_scalar_bc(solver.c_);
  solver.update_materials();
  solver.rho_previous_step_ = solver.rho_;
  solver.eta_previous_step_ = solver.eta_;
  solver.update_midpoint_materials();
  solver.pressure_.fill(0.0);
  solver.apply_scalar_bc(solver.pressure_);

  for (int i = 0; i < solver.u_.nx; ++i) {
    for (int j = 0; j < solver.u_.ny; ++j) {
      const double x = static_cast<double>(i) * solver.dx_;
      const double y = (static_cast<double>(j) + 0.5) * solver.dy_;
      solver.u_star_(i, j) = std::sin(2.0 * kPi * x) * std::sin(2.0 * kPi * y) + 0.2 * std::cos(2.0 * kPi * y);
    }
  }
  for (int i = 0; i < solver.v_.nx; ++i) {
    for (int j = 0; j < solver.v_.ny; ++j) {
      const double x = (static_cast<double>(i) + 0.5) * solver.dx_;
      const double y = static_cast<double>(j) * solver.dy_;
      solver.v_star_(i, j) = 0.5 * std::cos(2.0 * kPi * x) * std::sin(2.0 * kPi * y);
    }
  }
  solver.apply_u_bc(solver.u_star_);
  solver.apply_v_bc(solver.v_star_);

  const double div_before = divergence_l2(solver, solver.u_star_, solver.v_star_);
  const double residual = solver.solve_pressure_correction_jacobi();
  solver.apply_pressure_velocity_correction();
  const double div_after = divergence_l2(solver, solver.u_, solver.v_);

  require(std::isfinite(residual), "pressure residual must be finite");
  require(div_after < div_before * 1.0e-2, "variable-coefficient projection should strongly reduce divergence");
}

void test_ch_sparse_krylov_solver_recovers_reference_state() {
  ding::Config cfg = make_config();
  cfg.ch_inner_iterations = 200;
  cfg.ch_tolerance = 1.0e-12;

  ding::Solver solver(cfg);
  solver.initialize();

  ding::Field2D c_exact(cfg.nx, cfg.ny, cfg.ghost, 0.0);
  ding::Field2D lap(cfg.nx, cfg.ny, cfg.ghost, 0.0);
  ding::Field2D biharm(cfg.nx, cfg.ny, cfg.ghost, 0.0);
  ding::Field2D rhs(cfg.nx, cfg.ny, cfg.ghost, 0.0);

  const double alpha0 = 3.0 / (2.0 * cfg.dt);
  for (int i = 0; i < cfg.nx; ++i) {
    for (int j = 0; j < cfg.ny; ++j) {
      const double x = (static_cast<double>(i) + 0.5) * solver.dx_;
      const double y = (static_cast<double>(j) + 0.5) * solver.dy_;
      c_exact(i, j) = 0.42 + 0.08 * std::sin(2.0 * kPi * x) * std::cos(2.0 * kPi * y);
    }
  }
  solver.apply_scalar_bc(c_exact);

  for (int i = 0; i < cfg.nx; ++i) {
    for (int j = 0; j < cfg.ny; ++j) {
      lap(i, j) = solver.laplacian_center(c_exact, i, j);
    }
  }
  solver.apply_scalar_bc(lap);
  for (int i = 0; i < cfg.nx; ++i) {
    for (int j = 0; j < cfg.ny; ++j) {
      biharm(i, j) = solver.laplacian_center(lap, i, j);
      rhs(i, j) = alpha0 * c_exact(i, j) -
                  (cfg.stabilization_a1 * lap(i, j) - cfg.stabilization_a2 * biharm(i, j)) / cfg.pe;
    }
  }
  solver.apply_scalar_bc(biharm);
  solver.apply_scalar_bc(rhs);

  ding::Field2D c_solved(cfg.nx, cfg.ny, cfg.ghost, 0.0);
  double target_mean = 0.0;
  for (int i = 0; i < cfg.nx; ++i) {
    for (int j = 0; j < cfg.ny; ++j) {
      target_mean += c_exact(i, j);
    }
  }
  target_mean /= static_cast<double>(cfg.nx * cfg.ny);

  double iterate_residual = 0.0;
  double equation_residual = 0.0;
  solver.solve_phase_linear_system_eq25(rhs, alpha0, target_mean, c_solved, iterate_residual, equation_residual);

  double error_sq = 0.0;
  double norm_sq = 0.0;
  for (int i = 0; i < cfg.nx; ++i) {
    for (int j = 0; j < cfg.ny; ++j) {
      error_sq += std::pow(c_solved(i, j) - c_exact(i, j), 2);
      norm_sq += std::pow(c_exact(i, j), 2);
    }
  }
  const double relative_error = std::sqrt(error_sq / std::max(norm_sq, 1.0e-30));
  require(std::isfinite(iterate_residual), "ch iterate residual must be finite");
  require(equation_residual < 1.0e-10, "ch sparse Krylov solver should satisfy the linear system");
  require(relative_error < 1.0e-8, "ch sparse Krylov solver should recover the reference state");
}

void test_ch_step0_preserves_constant_equilibrium_state() {
  ding::Config cfg = make_config();
  cfg.dt = 1.0e-4;
  cfg.density_ratio = 1.0;
  cfg.viscosity_ratio = 1.0;

  ding::Solver solver(cfg);
  solver.initialize();

  for (int i = 0; i < cfg.nx; ++i) {
    for (int j = 0; j < cfg.ny; ++j) {
      solver.c_(i, j) = 1.0;
    }
  }
  solver.apply_scalar_bc(solver.c_);
  solver.c_previous_step_ = solver.c_;
  solver.c_two_steps_back_ = solver.c_;
  solver.update_chemical_potential(solver.c_, solver.mu_);

  ding::Field2D zero_u(solver.u_.nx, solver.u_.ny, solver.u_.ghost, 0.0);
  ding::Field2D zero_v(solver.v_.nx, solver.v_.ny, solver.v_.ghost, 0.0);
  solver.apply_u_bc(zero_u);
  solver.apply_v_bc(zero_v);

  const double residual = solver.solve_cahn_hilliard_semi_implicit(zero_u, zero_v, 0);
  require(std::isfinite(residual), "step-0 CH residual must be finite");

  double max_error = 0.0;
  for (int i = 0; i < cfg.nx; ++i) {
    for (int j = 0; j < cfg.ny; ++j) {
      max_error = std::max(max_error, std::abs(solver.c_(i, j) - 1.0));
    }
  }
  require(max_error < 1.0e-10, "step-0 CH startup should preserve a constant equilibrium state");
}

void test_advection_only_preserves_uniform_phase_field() {
  ding::Config cfg = make_config();
  cfg.mode = "advection_only";
  cfg.velocity_profile = "solid_body_rotation";
  cfg.angular_velocity = 2.0 * kPi;
  cfg.rotation_center_x = 0.5;
  cfg.rotation_center_y = 0.5;
  cfg.density_ratio = 1.0;
  cfg.viscosity_ratio = 1.0;

  ding::Solver solver(cfg);
  solver.initialize();

  for (int i = 0; i < cfg.nx; ++i) {
    for (int j = 0; j < cfg.ny; ++j) {
      solver.c_(i, j) = 0.3;
    }
  }
  solver.apply_scalar_bc(solver.c_);
  solver.c_previous_step_ = solver.c_;
  solver.c_two_steps_back_ = solver.c_;

  const bool ok = solver.advance_one_timestep(0);
  require(ok, "advection-only step must succeed");

  double max_error = 0.0;
  for (int i = 0; i < cfg.nx; ++i) {
    for (int j = 0; j < cfg.ny; ++j) {
      max_error = std::max(max_error, std::abs(solver.c_(i, j) - 0.3));
    }
  }
  require(max_error < 1.0e-12, "pure advection must preserve a uniform phase field");
  require(solver.last_ch_solver_name_ == "PureAdvectionSSPRK3[WENO5]",
          "advection-only mode should report the pure-advection solver");
}

void test_restart_snapshot_reproduces_reference_run() {
  namespace fs = std::filesystem;
  const fs::path base_dir = fs::path("/tmp") / "ding_nsch_core_restart_test";
  fs::remove_all(base_dir);

  ding::Config ref_cfg = make_config();
  ref_cfg.name = "reference";
  ref_cfg.output_dir = (base_dir / "reference").string();
  ref_cfg.steps = 6;
  ref_cfg.write_restart = true;
  ref_cfg.restart_every = 1;

  ding::Solver reference(ref_cfg);
  require(reference.run(), "reference run must succeed");

  ding::Config first_leg_cfg = ref_cfg;
  first_leg_cfg.name = "restart_case";
  first_leg_cfg.output_dir = (base_dir / "restart_case").string();
  first_leg_cfg.steps = 3;

  ding::Solver first_leg(first_leg_cfg);
  require(first_leg.run(), "initial restart-producing run must succeed");
  const fs::path restart_file = first_leg.restart_snapshot_path();
  require(fs::exists(restart_file), "restart snapshot must exist after checkpointed run");

  ding::Config resumed_cfg = ref_cfg;
  resumed_cfg.name = "restart_case";
  resumed_cfg.output_dir = (base_dir / "restart_case").string();
  resumed_cfg.restart = true;
  resumed_cfg.restart_file = restart_file.string();

  ding::Solver resumed(resumed_cfg);
  require(resumed.run(), "resumed run must succeed");

  require(resumed.history_.back().step == ref_cfg.steps, "resumed history must reach the target final step");
  require(resumed.history_.size() == static_cast<std::size_t>(ref_cfg.steps + 1),
          "resumed history must preserve pre-restart and post-restart entries");
  require(field_max_abs_difference(reference.c_, resumed.c_) < 1.0e-12, "restart must reproduce phase field");
  require(field_max_abs_difference(reference.u_, resumed.u_) < 1.0e-12, "restart must reproduce u velocity");
  require(field_max_abs_difference(reference.v_, resumed.v_) < 1.0e-12, "restart must reproduce v velocity");
  require(field_max_abs_difference(reference.pressure_, resumed.pressure_) < 1.0e-12,
          "restart must reproduce pressure");
}

} // namespace

int main() {
  try {
    test_midpoint_material_average();
    std::cout << "PASS test_midpoint_material_average\n";
    test_variable_coefficient_projection_reduces_divergence();
    std::cout << "PASS test_variable_coefficient_projection_reduces_divergence\n";
    test_ch_sparse_krylov_solver_recovers_reference_state();
    std::cout << "PASS test_ch_sparse_krylov_solver_recovers_reference_state\n";
    test_ch_step0_preserves_constant_equilibrium_state();
    std::cout << "PASS test_ch_step0_preserves_constant_equilibrium_state\n";
    test_advection_only_preserves_uniform_phase_field();
    std::cout << "PASS test_advection_only_preserves_uniform_phase_field\n";
    test_restart_snapshot_reproduces_reference_run();
    std::cout << "PASS test_restart_snapshot_reproduces_reference_run\n";
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "FAIL " << ex.what() << "\n";
    return 1;
  }
}
