#include "solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace ding {

namespace {

constexpr std::string_view kRestartMagic = "ding_nsch_restart_v1";

template <typename T>
void write_binary_value(std::ostream &out, const T &value) {
  out.write(reinterpret_cast<const char *>(&value), static_cast<std::streamsize>(sizeof(T)));
  if (!out) {
    throw std::runtime_error("failed to write restart snapshot");
  }
}

template <typename T>
T read_binary_value(std::istream &in) {
  T value{};
  in.read(reinterpret_cast<char *>(&value), static_cast<std::streamsize>(sizeof(T)));
  if (!in) {
    throw std::runtime_error("failed to read restart snapshot");
  }
  return value;
}

void write_binary_string(std::ostream &out, const std::string &value) {
  const std::uint64_t size = static_cast<std::uint64_t>(value.size());
  write_binary_value(out, size);
  out.write(value.data(), static_cast<std::streamsize>(value.size()));
  if (!out) {
    throw std::runtime_error("failed to write restart snapshot string");
  }
}

std::string read_binary_string(std::istream &in) {
  const std::uint64_t size = read_binary_value<std::uint64_t>(in);
  std::string value(size, '\0');
  in.read(value.data(), static_cast<std::streamsize>(size));
  if (!in) {
    throw std::runtime_error("failed to read restart snapshot string");
  }
  return value;
}

void write_binary_field(std::ostream &out, const Field2D &field) {
  write_binary_value(out, field.nx);
  write_binary_value(out, field.ny);
  write_binary_value(out, field.ghost);
  const std::uint64_t size = static_cast<std::uint64_t>(field.data.size());
  write_binary_value(out, size);
  out.write(reinterpret_cast<const char *>(field.data.data()),
            static_cast<std::streamsize>(size * sizeof(double)));
  if (!out) {
    throw std::runtime_error("failed to write restart snapshot field");
  }
}

void read_binary_field(std::istream &in, Field2D &field) {
  const int nx = read_binary_value<int>(in);
  const int ny = read_binary_value<int>(in);
  const int ghost = read_binary_value<int>(in);
  const std::uint64_t size = read_binary_value<std::uint64_t>(in);
  if (field.nx != nx || field.ny != ny || field.ghost != ghost || field.data.size() != size) {
    throw std::runtime_error("restart snapshot field size mismatch");
  }
  in.read(reinterpret_cast<char *>(field.data.data()), static_cast<std::streamsize>(size * sizeof(double)));
  if (!in) {
    throw std::runtime_error("failed to read restart snapshot field");
  }
}

bool nearly_equal(double a, double b) {
  const double scale = std::max({1.0, std::abs(a), std::abs(b)});
  return std::abs(a - b) <= 1.0e-12 * scale;
}

void require_compatible(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error("restart snapshot incompatible with config: " + message);
  }
}

std::vector<std::string> split_csv_row(const std::string &line) {
  std::vector<std::string> fields;
  std::string field;
  std::istringstream in(line);
  while (std::getline(in, field, ',')) {
    fields.push_back(field);
  }
  return fields;
}

} // namespace

std::string Solver::case_output_dir() const {
  namespace fs = std::filesystem;
  return (fs::path(cfg_.output_dir) / cfg_.name).string();
}

std::string Solver::case_log_path() const {
  namespace fs = std::filesystem;
  return (fs::path(case_output_dir()) / "run.log").string();
}

std::string Solver::history_csv_path() const {
  namespace fs = std::filesystem;
  return (fs::path(case_output_dir()) / "history.csv").string();
}

std::string Solver::pressure_solver_dir() const {
  namespace fs = std::filesystem;
  return (fs::path(case_output_dir()) / "pressure_solver").string();
}

std::string Solver::restart_snapshot_path() const {
  namespace fs = std::filesystem;
  return (fs::path(case_output_dir()) / "restart_latest.bin").string();
}

void Solver::open_case_log() {
  namespace fs = std::filesystem;
  const fs::path output_path = case_output_dir();
  fs::create_directories(output_path);
  const bool append_existing = (cfg_.restart || !cfg_.restart_file.empty()) && fs::exists(case_log_path());
  case_log_.open(case_log_path(), std::ios::out | (append_existing ? std::ios::app : std::ios::trunc));
  if (!case_log_) {
    throw std::runtime_error("cannot open run.log");
  }
}

void Solver::close_case_log() {
  if (case_log_.is_open()) {
    case_log_.flush();
    case_log_.close();
  }
}

void Solver::open_history_csv_stream() {
  namespace fs = std::filesystem;
  const fs::path output_path = case_output_dir();
  fs::create_directories(output_path);
  const bool append_existing = restarted_from_snapshot_ && fs::exists(history_csv_path());
  history_csv_stream_.open(history_csv_path(), std::ios::out | (append_existing ? std::ios::app : std::ios::trunc));
  if (!history_csv_stream_) {
    throw std::runtime_error("cannot open history.csv");
  }
  if (!append_existing) {
    write_history_csv_header(history_csv_stream_);
    history_csv_stream_.flush();
  }
}

void Solver::close_history_csv_stream() {
  if (history_csv_stream_.is_open()) {
    history_csv_stream_.flush();
    history_csv_stream_.close();
  }
}

void Solver::write_history_csv_header(std::ostream &out) const {
  out << "step,time,mass,mass_drift,divergence_l2,max_divergence,max_abs_mu,max_velocity,kinetic_energy,"
         "total_free_energy,ch_inner_residual,ch_equation_residual,coupling_residual,pressure_correction_residual,"
         "momentum_residual,boundary_speed_pre_correction,boundary_speed_post_correction,rho_min,rho_max,eta_min,"
         "eta_max,dt_limit_advective,dt_limit_capillary,dt_limit_ch_explicit,dt_limit_active,dt_limit_ratio,"
         "ch_iterations,coupling_iterations,pressure_iterations,momentum_iterations,"
         "ch_solver_name,momentum_solver_name,pressure_solver_name,dt_limit_source\n";
}

void Solver::append_history_csv_entry(int step, double time, const Diagnostics &diag) {
  if (!history_csv_stream_.is_open()) {
    return;
  }
  history_csv_stream_ << std::setprecision(17) << step << "," << time << "," << diag.mass << "," << diag.mass_drift
                      << "," << diag.divergence_l2 << "," << diag.max_divergence_after_correction << ","
                      << diag.max_abs_mu << "," << diag.max_velocity << "," << diag.kinetic_energy << ","
                      << diag.total_free_energy << "," << diag.ch_inner_residual << "," << diag.ch_equation_residual
                      << "," << diag.coupling_residual << "," << diag.pressure_correction_residual << ","
                      << diag.momentum_residual << "," << diag.boundary_speed_pre_correction << ","
                      << diag.boundary_speed_post_correction << "," << diag.rho_min << "," << diag.rho_max << ","
                      << diag.eta_min << "," << diag.eta_max << "," << diag.dt_limit_advective << ","
                      << diag.dt_limit_capillary << "," << diag.dt_limit_ch_explicit << ","
                      << diag.dt_limit_active << "," << diag.dt_limit_ratio << "," << diag.ch_iterations << ","
                      << diag.coupling_iterations << "," << diag.pressure_iterations << ","
                      << diag.momentum_iterations << "," << diag.ch_solver_name << "," << diag.momentum_solver_name
                      << "," << diag.pressure_solver_name << "," << diag.dt_limit_source << "\n";
  history_csv_stream_.flush();
}

void Solver::log_message(const std::string &message) {
  if (!case_log_.is_open()) {
    return;
  }
  case_log_ << message << "\n";
  case_log_.flush();
}

void Solver::log_run_header() {
  std::ostringstream out;
  out << std::setprecision(17);
  out << "CASE name=" << cfg_.name << " mode=" << cfg_.mode
      << " nx=" << cfg_.nx << " ny=" << cfg_.ny
      << " lx=" << cfg_.lx << " ly=" << cfg_.ly
      << " dt=" << cfg_.dt << " steps=" << cfg_.steps
      << " final_time=" << (cfg_.steps * cfg_.dt);
  log_message(out.str());

  out.str("");
  out.clear();
  out << "SOLVERS ch=SparsePCG pressure=" << cfg_.pressure_scheme
      << " momentum=ExplicitConvection+MonolithicImplicitViscosityBiCGSTAB"
      << " momentum_advection=" << cfg_.momentum_advection_scheme;
  log_message(out.str());

  out.str("");
  out.clear();
  out << std::setprecision(17)
      << "LIMITS ch_max_it=" << cfg_.ch_inner_iterations
      << " ch_tol=" << cfg_.ch_tolerance
      << " pressure_max_it=" << cfg_.poisson_iterations
      << " pressure_tol=" << cfg_.pressure_tolerance
      << " momentum_max_it=" << cfg_.momentum_iterations
      << " momentum_abs_tol=" << cfg_.momentum_tolerance
      << " coupling_max_it=" << cfg_.coupling_iterations
      << " coupling_tol=" << cfg_.coupling_tolerance;
  log_message(out.str());

  out.str("");
  out.clear();
  out << std::setprecision(17)
      << "PHYSICS re=" << cfg_.re << " ca=" << cfg_.ca << " pe=" << cfg_.pe << " cn=" << cfg_.cn
      << " density_ratio=" << cfg_.density_ratio
      << " viscosity_ratio=" << cfg_.viscosity_ratio
      << " periodic_x=" << cfg_.periodic_x
      << " periodic_y=" << cfg_.periodic_y
      << " top_wall_velocity_x=" << cfg_.top_wall_velocity_x
      << " bottom_wall_velocity_x=" << cfg_.bottom_wall_velocity_x
      << " restart=" << (cfg_.restart || !cfg_.restart_file.empty())
      << " write_restart=" << cfg_.write_restart
      << " restart_every=" << cfg_.restart_every;
  log_message(out.str());
}

bool Solver::should_write_restart(int step) const {
  if (!cfg_.write_restart) {
    return false;
  }
  const int interval = cfg_.restart_every > 0 ? cfg_.restart_every : std::max(1, cfg_.write_every);
  return step == 0 || step % interval == 0 || step == cfg_.steps;
}

void Solver::write_restart_snapshot(int step) const {
  namespace fs = std::filesystem;
  const fs::path final_path = restart_snapshot_path();
  fs::create_directories(final_path.parent_path());
  const fs::path temp_path = final_path.string() + ".tmp";

  std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("cannot open restart snapshot for writing");
  }

  write_binary_string(out, std::string(kRestartMagic));
  write_binary_value(out, static_cast<std::uint32_t>(1));
  write_binary_value(out, step);
  write_binary_value(out, initial_mass_);
  write_binary_string(out, cfg_.mode);
  write_binary_value(out, cfg_.nx);
  write_binary_value(out, cfg_.ny);
  write_binary_value(out, cfg_.ghost);
  write_binary_value(out, cfg_.dt);
  write_binary_value(out, cfg_.lx);
  write_binary_value(out, cfg_.ly);
  write_binary_value(out, cfg_.re);
  write_binary_value(out, cfg_.ca);
  write_binary_value(out, cfg_.pe);
  write_binary_value(out, cfg_.cn);
  write_binary_value(out, cfg_.density_ratio);
  write_binary_value(out, cfg_.viscosity_ratio);
  write_binary_value(out, cfg_.body_force_x);
  write_binary_value(out, cfg_.body_force_y);
  write_binary_value(out, cfg_.top_wall_velocity_x);
  write_binary_value(out, cfg_.bottom_wall_velocity_x);
  write_binary_value(out, static_cast<int>(cfg_.periodic_x));
  write_binary_value(out, static_cast<int>(cfg_.periodic_y));

  write_binary_field(out, c_);
  write_binary_field(out, c_previous_step_);
  write_binary_field(out, c_two_steps_back_);
  write_binary_field(out, pressure_);
  write_binary_field(out, pressure_previous_step_);
  write_binary_field(out, u_);
  write_binary_field(out, v_);
  write_binary_field(out, u_previous_step_);
  write_binary_field(out, v_previous_step_);
  write_binary_field(out, phase_explicit_operator_prev_);
  write_binary_field(out, momentum_u_rhs_prev_);
  write_binary_field(out, momentum_v_rhs_prev_);
  out.flush();
  if (!out) {
    throw std::runtime_error("failed to finalize restart snapshot");
  }
  out.close();
  if (!out) {
    throw std::runtime_error("failed to close restart snapshot");
  }
  fs::rename(temp_path, final_path);
}

void Solver::load_restart_snapshot() {
  namespace fs = std::filesystem;
  const fs::path restart_path = cfg_.restart_file.empty() ? fs::path(restart_snapshot_path()) : fs::path(cfg_.restart_file);
  std::ifstream in(restart_path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open restart snapshot: " + restart_path.string());
  }

  const std::string magic = read_binary_string(in);
  require_compatible(magic == kRestartMagic, "unrecognized restart file");
  const std::uint32_t version = read_binary_value<std::uint32_t>(in);
  require_compatible(version == 1u, "unsupported restart file version");

  restart_step_ = read_binary_value<int>(in);
  initial_mass_ = read_binary_value<double>(in);
  const std::string restart_mode = read_binary_string(in);
  require_compatible(restart_mode == cfg_.mode, "mode");
  require_compatible(read_binary_value<int>(in) == cfg_.nx, "nx");
  require_compatible(read_binary_value<int>(in) == cfg_.ny, "ny");
  require_compatible(read_binary_value<int>(in) == cfg_.ghost, "ghost");
  require_compatible(nearly_equal(read_binary_value<double>(in), cfg_.dt), "dt");
  require_compatible(nearly_equal(read_binary_value<double>(in), cfg_.lx), "lx");
  require_compatible(nearly_equal(read_binary_value<double>(in), cfg_.ly), "ly");
  require_compatible(nearly_equal(read_binary_value<double>(in), cfg_.re), "re");
  require_compatible(nearly_equal(read_binary_value<double>(in), cfg_.ca), "ca");
  require_compatible(nearly_equal(read_binary_value<double>(in), cfg_.pe), "pe");
  require_compatible(nearly_equal(read_binary_value<double>(in), cfg_.cn), "cn");
  require_compatible(nearly_equal(read_binary_value<double>(in), cfg_.density_ratio), "density_ratio");
  require_compatible(nearly_equal(read_binary_value<double>(in), cfg_.viscosity_ratio), "viscosity_ratio");
  require_compatible(nearly_equal(read_binary_value<double>(in), cfg_.body_force_x), "body_force_x");
  require_compatible(nearly_equal(read_binary_value<double>(in), cfg_.body_force_y), "body_force_y");
  require_compatible(nearly_equal(read_binary_value<double>(in), cfg_.top_wall_velocity_x), "top_wall_velocity_x");
  require_compatible(nearly_equal(read_binary_value<double>(in), cfg_.bottom_wall_velocity_x), "bottom_wall_velocity_x");
  require_compatible(read_binary_value<int>(in) == static_cast<int>(cfg_.periodic_x), "periodic_x");
  require_compatible(read_binary_value<int>(in) == static_cast<int>(cfg_.periodic_y), "periodic_y");

  read_binary_field(in, c_);
  read_binary_field(in, c_previous_step_);
  read_binary_field(in, c_two_steps_back_);
  read_binary_field(in, pressure_);
  read_binary_field(in, pressure_previous_step_);
  read_binary_field(in, u_);
  read_binary_field(in, v_);
  read_binary_field(in, u_previous_step_);
  read_binary_field(in, v_previous_step_);
  read_binary_field(in, phase_explicit_operator_prev_);
  read_binary_field(in, momentum_u_rhs_prev_);
  read_binary_field(in, momentum_v_rhs_prev_);

  restarted_from_snapshot_ = true;
  apply_scalar_bc(c_);
  apply_scalar_bc(c_previous_step_);
  apply_scalar_bc(c_two_steps_back_);
  apply_scalar_bc(pressure_);
  apply_scalar_bc(pressure_previous_step_);
  apply_u_velocity_bc(u_);
  apply_v_bc(v_);
  apply_u_velocity_bc(u_previous_step_);
  apply_v_bc(v_previous_step_);

  update_materials();
  update_materials_from_phase(c_previous_step_, rho_previous_step_, eta_previous_step_);
  update_midpoint_materials();
  update_chemical_potential(c_, mu_);

  pressure_correction_.fill(0.0);
  u_star_.fill(0.0);
  v_star_.fill(0.0);
  surface_fx_cell_.fill(0.0);
  surface_fy_cell_.fill(0.0);
  surface_fx_u_.fill(0.0);
  surface_fy_v_.fill(0.0);
  phase_advection_rhs_.fill(0.0);
  phase_advection_rhs_prev_.fill(0.0);
}

void Solver::load_history_csv() {
  namespace fs = std::filesystem;
  const fs::path history_path = history_csv_path();
  history_.clear();
  if (!fs::exists(history_path)) {
    return;
  }

  std::ifstream in(history_path);
  if (!in) {
    throw std::runtime_error("cannot open history.csv for restart");
  }

  std::string line;
  if (!std::getline(in, line)) {
    return;
  }

  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    const std::vector<std::string> values = split_csv_row(line);
    if (values.size() != 28 && values.size() != 34) {
      throw std::runtime_error("history.csv has unexpected column count");
    }

    HistoryEntry entry;
    Diagnostics diag;
    std::size_t idx = 0;
    entry.step = std::stoi(values[idx++]);
    entry.time = std::stod(values[idx++]);
    diag.mass = std::stod(values[idx++]);
    diag.mass_drift = std::stod(values[idx++]);
    diag.divergence_l2 = std::stod(values[idx++]);
    diag.max_divergence_after_correction = std::stod(values[idx++]);
    diag.max_abs_mu = std::stod(values[idx++]);
    diag.max_velocity = std::stod(values[idx++]);
    diag.kinetic_energy = std::stod(values[idx++]);
    diag.total_free_energy = std::stod(values[idx++]);
    diag.ch_inner_residual = std::stod(values[idx++]);
    diag.ch_equation_residual = std::stod(values[idx++]);
    diag.coupling_residual = std::stod(values[idx++]);
    diag.pressure_correction_residual = std::stod(values[idx++]);
    diag.momentum_residual = std::stod(values[idx++]);
    diag.boundary_speed_pre_correction = std::stod(values[idx++]);
    diag.boundary_speed_post_correction = std::stod(values[idx++]);
    diag.rho_min = std::stod(values[idx++]);
    diag.rho_max = std::stod(values[idx++]);
    diag.eta_min = std::stod(values[idx++]);
    diag.eta_max = std::stod(values[idx++]);
    if (values.size() == 34) {
      diag.dt_limit_advective = std::stod(values[idx++]);
      diag.dt_limit_capillary = std::stod(values[idx++]);
      diag.dt_limit_ch_explicit = std::stod(values[idx++]);
      diag.dt_limit_active = std::stod(values[idx++]);
      diag.dt_limit_ratio = std::stod(values[idx++]);
    }
    diag.ch_iterations = std::stoi(values[idx++]);
    diag.coupling_iterations = std::stoi(values[idx++]);
    diag.pressure_iterations = std::stoi(values[idx++]);
    diag.momentum_iterations = std::stoi(values[idx++]);
    diag.ch_solver_name = values[idx++];
    diag.momentum_solver_name = values[idx++];
    diag.pressure_solver_name = values[idx++];
    if (values.size() == 34) {
      diag.dt_limit_source = values[idx++];
    }
    entry.diag = diag;
    history_.push_back(entry);
  }
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
         "boundary_speed_post_correction,rho_min,rho_max,eta_min,eta_max,dt_limit_advective,dt_limit_capillary,"
         "dt_limit_ch_explicit,dt_limit_active,dt_limit_ratio,ch_iterations,coupling_iterations,"
         "pressure_iterations,momentum_iterations,ch_solver_name,momentum_solver_name,pressure_solver_name,"
         "dt_limit_source\n";
  out << std::setprecision(17) << cfg_.name << "," << cfg_.nx << "," << cfg_.ny << "," << cfg_.dt << ","
      << cfg_.steps << "," << cfg_.steps * cfg_.dt << "," << last_diag_.mass << "," << last_diag_.mass_drift << ","
      << last_diag_.divergence_l2 << "," << last_diag_.max_divergence_after_correction << ","
      << last_diag_.max_abs_mu << "," << last_diag_.max_velocity << "," << last_diag_.kinetic_energy << ","
      << last_diag_.total_free_energy << "," << last_diag_.ch_inner_residual << ","
      << last_diag_.ch_equation_residual << "," << last_diag_.coupling_residual << ","
      << last_diag_.pressure_correction_residual << "," << last_diag_.momentum_residual << ","
      << last_diag_.boundary_speed_pre_correction << "," << last_diag_.boundary_speed_post_correction << ","
      << last_diag_.rho_min << "," << last_diag_.rho_max << "," << last_diag_.eta_min << "," << last_diag_.eta_max
      << "," << last_diag_.dt_limit_advective << "," << last_diag_.dt_limit_capillary << ","
      << last_diag_.dt_limit_ch_explicit << "," << last_diag_.dt_limit_active << "," << last_diag_.dt_limit_ratio
      << "," << last_diag_.ch_iterations
      << "," << last_diag_.coupling_iterations << "," << last_diag_.pressure_iterations << ","
      << last_diag_.momentum_iterations << ","
      << last_diag_.ch_solver_name << "," << last_diag_.momentum_solver_name << ","
      << last_diag_.pressure_solver_name << "," << last_diag_.dt_limit_source << "\n";
}

void Solver::write_history_csv() const {
  namespace fs = std::filesystem;
  const fs::path output_path = case_output_dir();
  fs::create_directories(output_path);
  std::ofstream out((output_path / "history.csv").string());
  if (!out) {
    throw std::runtime_error("cannot open history.csv");
  }
  write_history_csv_header(out);
  out << std::setprecision(17);
  for (const HistoryEntry &entry : history_) {
    const Diagnostics &diag = entry.diag;
    out << entry.step << "," << entry.time << "," << diag.mass << "," << diag.mass_drift << "," << diag.divergence_l2
        << "," << diag.max_divergence_after_correction << "," << diag.max_abs_mu << "," << diag.max_velocity << ","
        << diag.kinetic_energy << "," << diag.total_free_energy << "," << diag.ch_inner_residual << ","
        << diag.ch_equation_residual << "," << diag.coupling_residual << "," << diag.pressure_correction_residual
        << "," << diag.momentum_residual << "," << diag.boundary_speed_pre_correction << ","
        << diag.boundary_speed_post_correction << "," << diag.rho_min << "," << diag.rho_max << "," << diag.eta_min
        << "," << diag.eta_max << "," << diag.dt_limit_advective << "," << diag.dt_limit_capillary << ","
        << diag.dt_limit_ch_explicit << "," << diag.dt_limit_active << "," << diag.dt_limit_ratio << ","
        << diag.ch_iterations << "," << diag.coupling_iterations << ","
        << diag.pressure_iterations << "," << diag.momentum_iterations << ","
        << diag.ch_solver_name << "," << diag.momentum_solver_name << "," << diag.pressure_solver_name << ","
        << diag.dt_limit_source << "\n";
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
