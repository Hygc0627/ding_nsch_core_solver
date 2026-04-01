#!/usr/bin/env python3

import argparse
import csv
import glob
import math
import os
import re


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

  if None in (nx, ny, origin_x, origin_y, spacing_x, spacing_y):
    raise RuntimeError(f"failed to parse VTK metadata from {path}")
  if len(values) != nx * ny:
    raise RuntimeError(f"phase field size mismatch in {path}: expected {nx * ny}, got {len(values)}")
  return nx, ny, origin_x, origin_y, spacing_x, spacing_y, values


def compute_centroid_y(vtk_path):
  nx, ny, ox, oy, dx, dy, values = read_phase_vtk(vtk_path)
  weights = [min(1.0, max(0.0, value)) for value in values]
  mass = sum(weights)
  if mass <= 1.0e-30:
    raise RuntimeError(f"phase mass vanished in {vtk_path}")
  yc = 0.0
  for j in range(ny):
    y = oy + j * dy
    row_offset = j * nx
    for i in range(nx):
      yc += weights[row_offset + i] * y
  return yc / mass


def load_history(path):
  rows = []
  with open(path, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
      rows.append({
          "step": int(row["step"]),
          "time": float(row["time"]),
          "max_velocity": float(row["max_velocity"]),
          "pressure_iterations": int(row["pressure_iterations"]),
          "pressure_residual": float(row["pressure_correction_residual"]),
          "div_l2": float(row["divergence_l2"]),
          "mass_drift": float(row["mass_drift"]),
      })
  if not rows:
    raise RuntimeError(f"empty history file: {path}")
  return rows


def load_centroids(case_dir, case_name, dt):
  step_pattern = re.compile(r"_step_(\d+)\.vtk$")
  rows = []
  for vtk_path in sorted(glob.glob(os.path.join(case_dir, f"{case_name}_step_*.vtk"))):
    match = step_pattern.search(vtk_path)
    if not match:
      continue
    step = int(match.group(1))
    rows.append({
        "step": step,
        "time": step * dt,
        "centroid_y": compute_centroid_y(vtk_path),
    })
  if not rows:
    raise RuntimeError(f"no VTK snapshots found in {case_dir}")
  return rows


def write_summary_csv(path, unsplit_history, split_history):
  unsplit_last = unsplit_history[-1]
  split_last = split_history[-1]
  with open(path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=[
        "case",
        "final_step",
        "final_time",
        "max_velocity",
        "pressure_iterations",
        "pressure_residual",
        "div_l2",
        "mass_drift",
    ])
    writer.writeheader()
    writer.writerow({
        "case": "unsplit_icpcg",
        "final_step": unsplit_last["step"],
        "final_time": unsplit_last["time"],
        "max_velocity": unsplit_last["max_velocity"],
        "pressure_iterations": unsplit_last["pressure_iterations"],
        "pressure_residual": unsplit_last["pressure_residual"],
        "div_l2": unsplit_last["div_l2"],
        "mass_drift": unsplit_last["mass_drift"],
    })
    writer.writerow({
        "case": "split_icpcg",
        "final_step": split_last["step"],
        "final_time": split_last["time"],
        "max_velocity": split_last["max_velocity"],
        "pressure_iterations": split_last["pressure_iterations"],
        "pressure_residual": split_last["pressure_residual"],
        "div_l2": split_last["div_l2"],
        "mass_drift": split_last["mass_drift"],
    })


def map_value(value, value_min, value_max, pixel_min, pixel_max):
  if abs(value_max - value_min) < 1.0e-30:
    return 0.5 * (pixel_min + pixel_max)
  return pixel_min + (value - value_min) / (value_max - value_min) * (pixel_max - pixel_min)


def add_axes(lines, x0, y0, width, height, title, xlabel, ylabel, x_ticks, y_ticks):
  lines.append(f'<text x="{x0 + width/2:.1f}" y="{y0 - 12:.1f}" text-anchor="middle" font-size="16">{title}</text>')
  lines.append(f'<line x1="{x0}" y1="{y0 + height}" x2="{x0 + width}" y2="{y0 + height}" stroke="black" stroke-width="1.3"/>')
  lines.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + height}" stroke="black" stroke-width="1.3"/>')
  lines.append(f'<text x="{x0 + width/2:.1f}" y="{y0 + height + 42:.1f}" text-anchor="middle" font-size="14">{xlabel}</text>')
  lines.append(
      f'<text x="{x0 - 46:.1f}" y="{y0 + height/2:.1f}" text-anchor="middle" font-size="14" '
      f'transform="rotate(-90 {x0 - 46:.1f} {y0 + height/2:.1f})">{ylabel}</text>'
  )
  for pos, label in x_ticks:
    lines.append(f'<line x1="{pos:.2f}" y1="{y0 + height}" x2="{pos:.2f}" y2="{y0 + height + 6}" stroke="black"/>')
    lines.append(f'<text x="{pos:.2f}" y="{y0 + height + 22:.2f}" text-anchor="middle" font-size="12">{label}</text>')
  for pos, label in y_ticks:
    lines.append(f'<line x1="{x0 - 6}" y1="{pos:.2f}" x2="{x0}" y2="{pos:.2f}" stroke="black"/>')
    lines.append(f'<text x="{x0 - 10:.2f}" y="{pos + 4:.2f}" text-anchor="end" font-size="12">{label}</text>')
    lines.append(f'<line x1="{x0}" y1="{pos:.2f}" x2="{x0 + width}" y2="{pos:.2f}" stroke="#dddddd" stroke-width="0.8"/>')


def write_svg(path, unsplit_centroids, split_centroids, unsplit_history, split_history):
  width = 980
  height = 920
  panel_width = 760
  panel_height = 260
  left = 140
  top1 = 60
  top2 = 390
  top3 = 690

  t_max = max(unsplit_history[-1]["time"], split_history[-1]["time"])
  x_ticks = []
  for idx in range(6):
    t = t_max * idx / 5.0
    x = map_value(t, 0.0, t_max, left, left + panel_width)
    x_ticks.append((x, f"{t:.2f}"))

  cy_values = [row["centroid_y"] for row in unsplit_centroids] + [row["centroid_y"] for row in split_centroids]
  cy_min = min(cy_values)
  cy_max = max(cy_values)
  cy_ticks = []
  for idx in range(6):
    value = cy_min + (cy_max - cy_min) * idx / 5.0
    y = map_value(value, cy_min, cy_max, top1 + panel_height, top1)
    cy_ticks.append((y, f"{value:.3f}"))

  vel_values = [row["max_velocity"] for row in unsplit_history] + [row["max_velocity"] for row in split_history]
  vel_min = 0.0
  vel_max = max(vel_values)
  vel_ticks = []
  for idx in range(6):
    value = vel_min + (vel_max - vel_min) * idx / 5.0
    y = map_value(value, vel_min, vel_max, top2 + panel_height, top2)
    vel_ticks.append((y, f"{value:.3f}"))

  pit_values = [row["pressure_iterations"] for row in unsplit_history] + [row["pressure_iterations"] for row in split_history]
  pit_min = min(pit_values)
  pit_max = max(pit_values)
  pit_ticks = []
  for idx in range(6):
    value = pit_min + (pit_max - pit_min) * idx / 5.0
    y = map_value(value, pit_min, pit_max, top3 + panel_height, top3)
    pit_ticks.append((y, f"{value:.0f}"))

  def polyline(rows, x_min, x_max, y_min, y_max, x0, y0):
    pts = []
    for row in rows:
      x = map_value(row["time"], x_min, x_max, x0, x0 + panel_width)
      key = "centroid_y" if "centroid_y" in row else "max_velocity" if "max_velocity" in row else "pressure_iterations"
      y = map_value(row[key], y_min, y_max, y0 + panel_height, y0)
      pts.append(f"{x:.2f},{y:.2f}")
    return " ".join(pts)

  cy_unsplit = polyline(unsplit_centroids, 0.0, t_max, cy_min, cy_max, left, top1)
  cy_split = polyline(split_centroids, 0.0, t_max, cy_min, cy_max, left, top1)
  vel_unsplit = polyline(unsplit_history, 0.0, t_max, vel_min, vel_max, left, top2)
  vel_split = polyline(split_history, 0.0, t_max, vel_min, vel_max, left, top2)
  pit_unsplit = polyline(unsplit_history, 0.0, t_max, pit_min, pit_max, left, top3)
  pit_split = polyline(split_history, 0.0, t_max, pit_min, pit_max, left, top3)

  lines = [
      f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
      '<rect width="100%" height="100%" fill="white"/>',
      '<text x="490" y="28" text-anchor="middle" font-size="20">Rising bubble: split vs non-split pressure projection</text>',
  ]
  add_axes(lines, left, top1, panel_width, panel_height, "Bubble centroid height", "time", "centroid_y", x_ticks, cy_ticks)
  add_axes(lines, left, top2, panel_width, panel_height, "Maximum speed", "time", "max|u|", x_ticks, vel_ticks)
  add_axes(lines, left, top3, panel_width, panel_height, "Pressure iterations per step", "time", "p_it", x_ticks, pit_ticks)

  lines.extend([
      f'<polyline fill="none" stroke="#005f73" stroke-width="2.2" points="{cy_unsplit}"/>',
      f'<polyline fill="none" stroke="#bb3e03" stroke-width="2.2" points="{cy_split}"/>',
      f'<polyline fill="none" stroke="#005f73" stroke-width="2.2" points="{vel_unsplit}"/>',
      f'<polyline fill="none" stroke="#bb3e03" stroke-width="2.2" points="{vel_split}"/>',
      f'<polyline fill="none" stroke="#005f73" stroke-width="2.2" points="{pit_unsplit}"/>',
      f'<polyline fill="none" stroke="#bb3e03" stroke-width="2.2" points="{pit_split}"/>',
      '<rect x="690" y="42" width="220" height="58" fill="white" stroke="#bbbbbb"/>',
      '<line x1="710" y1="64" x2="760" y2="64" stroke="#005f73" stroke-width="2.2"/>',
      '<text x="772" y="68" font-size="13">icpcg (non-split)</text>',
      '<line x1="710" y1="86" x2="760" y2="86" stroke="#bb3e03" stroke-width="2.2"/>',
      '<text x="772" y="90" font-size="13">paper_split_icpcg</text>',
      '</svg>',
  ])

  with open(path, "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines))


def main():
  parser = argparse.ArgumentParser(description="Compare rising-bubble runs with split and non-split pressure projection.")
  parser.add_argument("--unsplit-dir", required=True)
  parser.add_argument("--unsplit-name", required=True)
  parser.add_argument("--split-dir", required=True)
  parser.add_argument("--split-name", required=True)
  parser.add_argument("--dt", type=float, required=True)
  parser.add_argument("--output-dir", required=True)
  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)

  unsplit_history = load_history(os.path.join(args.unsplit_dir, "history.csv"))
  split_history = load_history(os.path.join(args.split_dir, "history.csv"))
  unsplit_centroids = load_centroids(args.unsplit_dir, args.unsplit_name, args.dt)
  split_centroids = load_centroids(args.split_dir, args.split_name, args.dt)

  summary_csv = os.path.join(args.output_dir, "rising_bubble_split_vs_unsplit_summary.csv")
  write_summary_csv(summary_csv, unsplit_history, split_history)

  centroid_csv = os.path.join(args.output_dir, "rising_bubble_split_vs_unsplit_centroids.csv")
  with open(centroid_csv, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["case", "step", "time", "centroid_y"])
    writer.writeheader()
    for row in unsplit_centroids:
      writer.writerow({"case": "unsplit_icpcg", **row})
    for row in split_centroids:
      writer.writerow({"case": "split_icpcg", **row})

  svg_path = os.path.join(args.output_dir, "rising_bubble_split_vs_unsplit.svg")
  write_svg(svg_path, unsplit_centroids, split_centroids, unsplit_history, split_history)

  print(f"COMPARE_OK summary={summary_csv} centroids={centroid_csv} svg={svg_path}")


if __name__ == "__main__":
  main()
