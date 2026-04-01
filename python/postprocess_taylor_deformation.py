#!/usr/bin/env python3

import argparse
import csv
import glob
import math
import os
import re


def load_config(path):
  kv = {}
  with open(path, "r", encoding="utf-8") as handle:
    for raw_line in handle:
      line = raw_line.split("#", 1)[0].strip()
      if not line or "=" not in line:
        continue
      key, value = line.split("=", 1)
      kv[key.strip()] = value.strip()
  return kv


def read_phase_vtk(path):
  nx = ny = None
  origin_x = origin_y = None
  spacing_x = spacing_y = None
  values = []
  reading_phase = False
  skip_lookup = False

  with open(path, "r", encoding="utf-8") as handle:
    for raw_line in handle:
      line = raw_line.strip()
      if line.startswith("DIMENSIONS"):
        parts = line.split()
        nx = int(parts[1])
        ny = int(parts[2])
      elif line.startswith("ORIGIN"):
        parts = line.split()
        origin_x = float(parts[1])
        origin_y = float(parts[2])
      elif line.startswith("SPACING"):
        parts = line.split()
        spacing_x = float(parts[1])
        spacing_y = float(parts[2])
      elif line.startswith("SCALARS phase_fraction"):
        reading_phase = True
        skip_lookup = True
      elif reading_phase and skip_lookup:
        skip_lookup = False
      elif reading_phase:
        if line.startswith("SCALARS ") or line.startswith("VECTORS "):
          break
        if line:
          values.extend(float(item) for item in line.split())

  if nx is None or ny is None or origin_x is None or origin_y is None or spacing_x is None or spacing_y is None:
    raise RuntimeError(f"failed to parse VTK grid metadata from {path}")
  if len(values) != nx * ny:
    raise RuntimeError(f"phase field size mismatch in {path}: expected {nx * ny}, got {len(values)}")

  return nx, ny, origin_x, origin_y, spacing_x, spacing_y, values


def compute_moment_taylor_ratio(nx, ny, origin_x, origin_y, spacing_x, spacing_y, phase_values):
  weights = [min(1.0, max(0.0, value)) for value in phase_values]
  mass = sum(weights)
  if mass <= 1.0e-30:
    raise RuntimeError("droplet mass vanished during post-processing")

  xc = 0.0
  yc = 0.0
  for j in range(ny):
    y = origin_y + j * spacing_y
    for i in range(nx):
      x = origin_x + i * spacing_x
      w = weights[j * nx + i]
      xc += w * x
      yc += w * y
  xc /= mass
  yc /= mass

  mxx = 0.0
  myy = 0.0
  mxy = 0.0
  for j in range(ny):
    y = origin_y + j * spacing_y
    for i in range(nx):
      x = origin_x + i * spacing_x
      w = weights[j * nx + i]
      dx = x - xc
      dy = y - yc
      mxx += w * dx * dx
      myy += w * dy * dy
      mxy += w * dx * dy
  mxx /= mass
  myy /= mass
  mxy /= mass

  trace = mxx + myy
  det = mxx * myy - mxy * mxy
  disc = math.sqrt(max(0.0, 0.25 * trace * trace - det))
  eig_major = max(0.0, 0.5 * trace + disc)
  eig_minor = max(0.0, 0.5 * trace - disc)
  semi_major = 2.0 * math.sqrt(eig_major)
  semi_minor = 2.0 * math.sqrt(eig_minor)
  deformation = 0.0
  if semi_major + semi_minor > 1.0e-30:
    deformation = (semi_major - semi_minor) / (semi_major + semi_minor)
  angle = 0.5 * math.atan2(2.0 * mxy, mxx - myy)
  return deformation, semi_major, semi_minor, angle, xc, yc


def write_svg_plot(rows, path):
  width = 860
  height = 520
  margin_left = 90
  margin_right = 30
  margin_top = 30
  margin_bottom = 70
  plot_width = width - margin_left - margin_right
  plot_height = height - margin_top - margin_bottom

  x_values = [row["shear_time"] for row in rows]
  y_values = [row["taylor_deformation"] for row in rows]
  x_min = min(x_values)
  x_max = max(x_values)
  y_min = min(y_values)
  y_max = max(y_values)
  if abs(x_max - x_min) < 1.0e-30:
    x_max = x_min + 1.0
  if abs(y_max - y_min) < 1.0e-30:
    y_max = y_min + 1.0e-6

  def map_x(value):
    return margin_left + (value - x_min) / (x_max - x_min) * plot_width

  def map_y(value):
    return margin_top + plot_height - (value - y_min) / (y_max - y_min) * plot_height

  polyline = " ".join(f"{map_x(x):.2f},{map_y(y):.2f}" for x, y in zip(x_values, y_values))
  y_ticks = 5
  x_ticks = 5

  lines = [
      f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
      '<rect width="100%" height="100%" fill="white"/>',
      f'<text x="{width / 2:.1f}" y="20" text-anchor="middle" font-size="18">Taylor deformation in shear flow</text>',
      f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="black" stroke-width="1.5"/>',
      f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="black" stroke-width="1.5"/>',
      f'<polyline fill="none" stroke="#005f73" stroke-width="2" points="{polyline}"/>',
      f'<text x="{width / 2:.1f}" y="{height - 20}" text-anchor="middle" font-size="16">shear time (gamma_dot t)</text>',
      f'<text x="22" y="{height / 2:.1f}" text-anchor="middle" font-size="16" transform="rotate(-90 22 {height / 2:.1f})">Taylor deformation D</text>',
  ]

  for idx in range(x_ticks + 1):
    value = x_min + (x_max - x_min) * idx / x_ticks
    x = map_x(value)
    lines.append(f'<line x1="{x:.2f}" y1="{margin_top + plot_height}" x2="{x:.2f}" y2="{margin_top + plot_height + 6}" stroke="black"/>')
    lines.append(f'<text x="{x:.2f}" y="{margin_top + plot_height + 24}" text-anchor="middle" font-size="13">{value:.3f}</text>')

  for idx in range(y_ticks + 1):
    value = y_min + (y_max - y_min) * idx / y_ticks
    y = map_y(value)
    lines.append(f'<line x1="{margin_left - 6}" y1="{y:.2f}" x2="{margin_left}" y2="{y:.2f}" stroke="black"/>')
    lines.append(f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="13">{value:.4f}</text>')
    lines.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#cccccc" stroke-width="0.8"/>')

  lines.append("</svg>")
  with open(path, "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines))


def main():
  parser = argparse.ArgumentParser(description="Post-process Taylor deformation ratio from VTK snapshots.")
  parser.add_argument("--config", required=True, help="Path to the case config file.")
  parser.add_argument("--case-dir", default="", help="Optional explicit case output directory.")
  args = parser.parse_args()

  cfg = load_config(args.config)
  case_dir = args.case_dir or os.path.join(cfg.get("output_dir", "output"), cfg["name"])
  vtk_paths = sorted(glob.glob(os.path.join(case_dir, f"{cfg['name']}_step_*.vtk")))
  if not vtk_paths:
    raise RuntimeError(f"no VTK snapshots found in {case_dir}")

  dt = float(cfg["dt"])
  ly = float(cfg["ly"])
  top_wall = float(cfg.get("top_wall_velocity_x", "0.0"))
  bottom_wall = float(cfg.get("bottom_wall_velocity_x", "0.0"))
  shear_rate = (top_wall - bottom_wall) / max(ly, 1.0e-30)
  step_pattern = re.compile(r"_step_(\d+)\.vtk$")

  rows = []
  for vtk_path in vtk_paths:
    match = step_pattern.search(vtk_path)
    if not match:
      continue
    step = int(match.group(1))
    nx, ny, origin_x, origin_y, spacing_x, spacing_y, phase_values = read_phase_vtk(vtk_path)
    deformation, semi_major, semi_minor, angle, xc, yc = compute_moment_taylor_ratio(
        nx, ny, origin_x, origin_y, spacing_x, spacing_y, phase_values
    )
    time_value = step * dt
    rows.append({
        "step": step,
        "time": time_value,
        "shear_time": shear_rate * time_value,
        "taylor_deformation": deformation,
        "semi_major": semi_major,
        "semi_minor": semi_minor,
        "angle_rad": angle,
        "centroid_x": xc,
        "centroid_y": yc,
    })

  csv_path = os.path.join(case_dir, "taylor_deformation.csv")
  with open(csv_path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

  svg_path = os.path.join(case_dir, "taylor_deformation.svg")
  write_svg_plot(rows, svg_path)

  print(f"POSTPROCESS_OK case_dir={case_dir} csv={csv_path} svg={svg_path} frames={len(rows)}")


if __name__ == "__main__":
  main()
