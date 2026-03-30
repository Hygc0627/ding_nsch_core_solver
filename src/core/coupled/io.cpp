#include "solver.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace ding {

std::string Solver::case_output_dir() const {
  namespace fs = std::filesystem;
  return (fs::path(cfg_.output_dir) / cfg_.name).string();
}

std::string Solver::pressure_solver_dir() const {
  namespace fs = std::filesystem;
  return (fs::path(case_output_dir()) / "pressure_solver").string();
}

void Solver::write_visualization(int step) const {
  namespace fs = std::filesystem;
  const fs::path output_path = case_output_dir();
  fs::create_directories(output_path);

  std::ostringstream path;
  path << output_path.string() << "/" << cfg_.name << "_step_" << std::setw(6) << std::setfill('0') << step << ".vtk";
  std::ofstream out(path.str());
  if (!out) {
    throw std::runtime_error("cannot open visualization file: " + path.str());
  }

  out << "# vtk DataFile Version 3.0\n";
  out << "ding_nsch fields\n";
  out << "ASCII\n";
  out << "DATASET STRUCTURED_POINTS\n";
  out << "DIMENSIONS " << cfg_.nx << " " << cfg_.ny << " 1\n";
  out << "ORIGIN " << 0.5 * dx_ << " " << 0.5 * dy_ << " 0\n";
  out << "SPACING " << dx_ << " " << dy_ << " 1\n";
  out << "POINT_DATA " << cfg_.nx * cfg_.ny << "\n";

  out << "SCALARS phase_fraction double 1\n";
  out << "LOOKUP_TABLE default\n";
  for (int j = 0; j < cfg_.ny; ++j) {
    for (int i = 0; i < cfg_.nx; ++i) {
      out << c_(i, j) << "\n";
    }
  }

  out << "SCALARS pressure double 1\n";
  out << "LOOKUP_TABLE default\n";
  for (int j = 0; j < cfg_.ny; ++j) {
    for (int i = 0; i < cfg_.nx; ++i) {
      out << pressure_(i, j) << "\n";
    }
  }

  out << "SCALARS chemical_potential double 1\n";
  out << "LOOKUP_TABLE default\n";
  for (int j = 0; j < cfg_.ny; ++j) {
    for (int i = 0; i < cfg_.nx; ++i) {
      out << mu_(i, j) << "\n";
    }
  }

  out << "VECTORS velocity double\n";
  for (int j = 0; j < cfg_.ny; ++j) {
    for (int i = 0; i < cfg_.nx; ++i) {
      out << cell_centered_u(u_, i, j) << " " << cell_centered_v(v_, i, j) << " 0\n";
    }
  }

  write_pvd_index();
}

void Solver::write_pvd_index() const {
  namespace fs = std::filesystem;
  const fs::path output_path = case_output_dir();
  fs::create_directories(output_path);

  std::ofstream out((output_path / (cfg_.name + ".pvd")).string());
  if (!out) {
    throw std::runtime_error("cannot open PVD index file");
  }

  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  out << "  <Collection>\n";

  auto write_frame = [&](int step) {
    std::ostringstream vtk_name;
    vtk_name << cfg_.name << "_step_" << std::setw(6) << std::setfill('0') << step << ".vtk";
    out << "    <DataSet timestep=\"" << std::fixed << std::setprecision(12) << step * cfg_.dt
        << "\" group=\"\" part=\"0\" file=\"" << vtk_name.str() << "\"/>\n";
  };

  write_frame(0);
  for (int step = 1; step <= cfg_.steps; ++step) {
    if (step % cfg_.write_every == 0 || step == cfg_.steps) {
      write_frame(step);
    }
  }

  out << "  </Collection>\n";
  out << "</VTKFile>\n";
}

void Solver::write_summary_csv() const {
  namespace fs = std::filesystem;
  const fs::path output_path = case_output_dir();
  fs::create_directories(output_path);
  std::ofstream out((output_path / "summary.csv").string());
  if (!out) {
    throw std::runtime_error("cannot open summary.csv");
  }
  out << "name,nx,ny,dt,steps,final_time,mass,mass_drift,divergence_l2,max_divergence,max_abs_mu,max_velocity,"
         "kinetic_energy,total_free_energy,ch_inner_residual,ch_equation_residual,coupling_residual,"
         "pressure_correction_residual,momentum_residual,boundary_speed_pre_correction,"
         "boundary_speed_post_correction,rho_min,rho_max,eta_min,eta_max,coupling_iterations,"
         "pressure_iterations,momentum_iterations\n";
  out << std::setprecision(17) << cfg_.name << "," << cfg_.nx << "," << cfg_.ny << "," << cfg_.dt << ","
      << cfg_.steps << "," << cfg_.steps * cfg_.dt << "," << last_diag_.mass << "," << last_diag_.mass_drift << ","
      << last_diag_.divergence_l2 << "," << last_diag_.max_divergence_after_correction << ","
      << last_diag_.max_abs_mu << "," << last_diag_.max_velocity << "," << last_diag_.kinetic_energy << ","
      << last_diag_.total_free_energy << "," << last_diag_.ch_inner_residual << ","
      << last_diag_.ch_equation_residual << "," << last_diag_.coupling_residual << ","
      << last_diag_.pressure_correction_residual << "," << last_diag_.momentum_residual << ","
      << last_diag_.boundary_speed_pre_correction << "," << last_diag_.boundary_speed_post_correction << ","
      << last_diag_.rho_min << "," << last_diag_.rho_max << "," << last_diag_.eta_min << "," << last_diag_.eta_max
      << "," << last_diag_.coupling_iterations << "," << last_diag_.pressure_iterations << ","
      << last_diag_.momentum_iterations << "\n";
}

void Solver::write_history_csv() const {
  namespace fs = std::filesystem;
  const fs::path output_path = case_output_dir();
  fs::create_directories(output_path);
  std::ofstream out((output_path / "history.csv").string());
  if (!out) {
    throw std::runtime_error("cannot open history.csv");
  }
  out << "step,time,mass,mass_drift,divergence_l2,max_divergence,max_abs_mu,max_velocity,kinetic_energy,"
         "total_free_energy,ch_inner_residual,ch_equation_residual,coupling_residual,pressure_correction_residual,"
         "momentum_residual,boundary_speed_pre_correction,boundary_speed_post_correction,rho_min,rho_max,eta_min,"
         "eta_max,coupling_iterations,pressure_iterations,momentum_iterations\n";
  out << std::setprecision(17);
  for (const HistoryEntry &entry : history_) {
    const Diagnostics &diag = entry.diag;
    out << entry.step << "," << entry.time << "," << diag.mass << "," << diag.mass_drift << "," << diag.divergence_l2
        << "," << diag.max_divergence_after_correction << "," << diag.max_abs_mu << "," << diag.max_velocity << ","
        << diag.kinetic_energy << "," << diag.total_free_energy << "," << diag.ch_inner_residual << ","
        << diag.ch_equation_residual << "," << diag.coupling_residual << "," << diag.pressure_correction_residual
        << "," << diag.momentum_residual << "," << diag.boundary_speed_pre_correction << ","
        << diag.boundary_speed_post_correction << "," << diag.rho_min << "," << diag.rho_max << "," << diag.eta_min
        << "," << diag.eta_max << "," << diag.coupling_iterations << "," << diag.pressure_iterations << ","
        << diag.momentum_iterations << "\n";
  }
}

void Solver::write_final_cell_fields_csv() const {
  namespace fs = std::filesystem;
  const fs::path output_path = case_output_dir();
  fs::create_directories(output_path);
  std::ofstream out((output_path / "final_cell_fields.csv").string());
  if (!out) {
    throw std::runtime_error("cannot open final_cell_fields.csv");
  }
  out << "i,j,x,y,c,rho,eta,pressure,chemical_potential,u,v,divergence\n";
  out << std::setprecision(17);
  for (int j = 0; j < cfg_.ny; ++j) {
    for (int i = 0; i < cfg_.nx; ++i) {
      const double x = (static_cast<double>(i) + 0.5) * dx_;
      const double y = (static_cast<double>(j) + 0.5) * dy_;
      out << i << "," << j << "," << x << "," << y << "," << c_(i, j) << "," << rho_(i, j) << "," << eta_(i, j)
          << "," << pressure_(i, j) << "," << mu_(i, j) << "," << cell_centered_u(u_, i, j) << ","
          << cell_centered_v(v_, i, j) << "," << divergence_cell(u_, v_, i, j) << "\n";
    }
  }
}


} // namespace ding
