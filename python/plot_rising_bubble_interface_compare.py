#!/usr/bin/env python3

import argparse
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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

  x = origin_x + spacing_x * np.arange(nx)
  y = origin_y + spacing_y * np.arange(ny)
  field = np.array(values, dtype=float).reshape((ny, nx))
  return x, y, field


def make_path(case_dir, case_name, step):
  return Path(case_dir) / f"{case_name}_step_{step:06d}.vtk"


def main():
  parser = argparse.ArgumentParser(description="Overlay rising-bubble interfaces for split/non-split pressure runs.")
  parser.add_argument("--unsplit-dir", required=True)
  parser.add_argument("--unsplit-name", required=True)
  parser.add_argument("--split-dir", required=True)
  parser.add_argument("--split-name", required=True)
  parser.add_argument("--steps", nargs="+", type=int, required=True)
  parser.add_argument("--dt", type=float, required=True)
  parser.add_argument("--output", required=True)
  args = parser.parse_args()

  ncols = len(args.steps)
  fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 5.6), constrained_layout=True)
  if ncols == 1:
    axes = [axes]

  for ax, step in zip(axes, args.steps):
    unsplit_path = make_path(args.unsplit_dir, args.unsplit_name, step)
    split_path = make_path(args.split_dir, args.split_name, step)
    x_u, y_u, c_u = read_phase_vtk(unsplit_path)
    x_s, y_s, c_s = read_phase_vtk(split_path)

    ax.contour(x_u, y_u, c_u, levels=[0.5], colors=["#005f73"], linewidths=2.2)
    ax.contour(x_s, y_s, c_s, levels=[0.5], colors=["#bb3e03"], linewidths=2.0, linestyles=["--"])
    ax.set_aspect("equal")
    ax.set_xlim(float(x_u[0]), float(x_u[-1]))
    ax.set_ylim(float(y_u[0]), float(y_u[-1]))
    ax.grid(alpha=0.18)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"t = {step * args.dt:.3f}")

  fig.suptitle("Rising bubble interface: non-split vs split pressure projection", fontsize=16)
  handles = [
      plt.Line2D([0], [0], color="#005f73", lw=2.2, label="icpcg"),
      plt.Line2D([0], [0], color="#bb3e03", lw=2.0, ls="--", label="paper_split_icpcg"),
  ]
  fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.98))
  fig.savefig(args.output, format="svg")
  print(f"INTERFACE_COMPARE_OK output={args.output}")


if __name__ == "__main__":
  main()
