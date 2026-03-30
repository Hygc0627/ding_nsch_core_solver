#include <cstddef>
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
  solver.solve_phase_linear_system_eq25(rhs, target_mean, c_solved, iterate_residual, equation_residual);

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

} // namespace

int main() {
  try {
    test_midpoint_material_average();
    std::cout << "PASS test_midpoint_material_average\n";
    test_variable_coefficient_projection_reduces_divergence();
    std::cout << "PASS test_variable_coefficient_projection_reduces_divergence\n";
    test_ch_sparse_krylov_solver_recovers_reference_state();
    std::cout << "PASS test_ch_sparse_krylov_solver_recovers_reference_state\n";
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "FAIL " << ex.what() << "\n";
    return 1;
  }
}
