#!/usr/bin/env python3

import argparse
from pathlib import Path


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

  x = [origin_x + spacing_x * i for i in range(nx)]
  y = [origin_y + spacing_y * j for j in range(ny)]
  field = []
  for j in range(ny):
    row = values[j * nx:(j + 1) * nx]
    field.append(row)
  return x, y, field


def interpolate(p1, v1, p2, v2, level):
  if abs(v2 - v1) < 1.0e-30:
    t = 0.5
  else:
    t = (level - v1) / (v2 - v1)
  return p1 + t * (p2 - p1)


def contour_segments(x, y, field, level=0.5):
  nx = len(x)
  ny = len(y)
  segments = []

  def edge_point(edge, x0, y0, x1, y1, f0, f1, f2, f3):
    if edge == 0:
      return interpolate(x0, f0, x1, f1, level), y0
    if edge == 1:
      return x1, interpolate(y0, f1, y1, f2, level)
    if edge == 2:
      return interpolate(x0, f3, x1, f2, level), y1
    if edge == 3:
      return x0, interpolate(y0, f0, y1, f3, level)
    raise ValueError("invalid edge")

  case_table = {
      0: [],
      1: [(3, 0)],
      2: [(0, 1)],
      3: [(3, 1)],
      4: [(1, 2)],
      5: [(3, 2), (0, 1)],
      6: [(0, 2)],
      7: [(3, 2)],
      8: [(2, 3)],
      9: [(0, 2)],
      10: [(0, 3), (1, 2)],
      11: [(1, 2)],
      12: [(1, 3)],
      13: [(0, 1)],
      14: [(3, 0)],
      15: [],
  }

  for j in range(ny - 1):
    for i in range(nx - 1):
      x0 = x[i]
      x1 = x[i + 1]
      y0 = y[j]
      y1 = y[j + 1]
      f0 = field[j][i]
      f1 = field[j][i + 1]
      f2 = field[j + 1][i + 1]
      f3 = field[j + 1][i]
      mask = 0
      if f0 >= level:
        mask |= 1
      if f1 >= level:
        mask |= 2
      if f2 >= level:
        mask |= 4
      if f3 >= level:
        mask |= 8
      for e0, e1 in case_table[mask]:
        p0 = edge_point(e0, x0, y0, x1, y1, f0, f1, f2, f3)
        p1 = edge_point(e1, x0, y0, x1, y1, f0, f1, f2, f3)
        segments.append((p0, p1))
  return segments


def map_x(x, xmin, xmax, left, width):
  return left + (x - xmin) / max(xmax - xmin, 1.0e-30) * width


def map_y(y, ymin, ymax, top, height):
  return top + height - (y - ymin) / max(ymax - ymin, 1.0e-30) * height


def add_panel(lines, x, y, field_a, field_b, left, top, width, height, title):
  xmin = x[0]
  xmax = x[-1]
  ymin = y[0]
  ymax = y[-1]
  segs_a = contour_segments(x, y, field_a, 0.5)
  segs_b = contour_segments(x, y, field_b, 0.5)

  lines.append(f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="white" stroke="#999999"/>')
  for gx in range(6):
    xg = left + gx * width / 5.0
    lines.append(f'<line x1="{xg:.2f}" y1="{top}" x2="{xg:.2f}" y2="{top + height}" stroke="#eeeeee"/>')
  for gy in range(6):
    yg = top + gy * height / 5.0
    lines.append(f'<line x1="{left}" y1="{yg:.2f}" x2="{left + width}" y2="{yg:.2f}" stroke="#eeeeee"/>')
  lines.append(f'<text x="{left + width/2:.1f}" y="{top - 12:.1f}" text-anchor="middle" font-size="16">{title}</text>')

  for (p0, p1) in segs_a:
    x0 = map_x(p0[0], xmin, xmax, left, width)
    y0 = map_y(p0[1], ymin, ymax, top, height)
    x1 = map_x(p1[0], xmin, xmax, left, width)
    y1 = map_y(p1[1], ymin, ymax, top, height)
    lines.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" stroke="#005f73" stroke-width="2.0"/>')

  for (p0, p1) in segs_b:
    x0 = map_x(p0[0], xmin, xmax, left, width)
    y0 = map_y(p0[1], ymin, ymax, top, height)
    x1 = map_x(p1[0], xmin, xmax, left, width)
    y1 = map_y(p1[1], ymin, ymax, top, height)
    lines.append(
        f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
        'stroke="#bb3e03" stroke-width="1.8" stroke-dasharray="6 4"/>'
    )


def main():
  parser = argparse.ArgumentParser(description="Make a pure-SVG rising-bubble interface comparison.")
  parser.add_argument("--unsplit-dir", required=True)
  parser.add_argument("--unsplit-name", required=True)
  parser.add_argument("--split-dir", required=True)
  parser.add_argument("--split-name", required=True)
  parser.add_argument("--steps", nargs="+", type=int, required=True)
  parser.add_argument("--dt", type=float, required=True)
  parser.add_argument("--output", required=True)
  args = parser.parse_args()

  width = 420 * len(args.steps)
  height = 520
  panel_w = 320
  panel_h = 400
  top = 80
  left0 = 60
  gap = 40

  lines = [
      f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
      '<rect width="100%" height="100%" fill="white"/>',
      f'<text x="{width/2:.1f}" y="30" text-anchor="middle" font-size="20">Rising bubble interface: split vs non-split pressure projection</text>',
      '<line x1="80" y1="52" x2="130" y2="52" stroke="#005f73" stroke-width="2.0"/>',
      '<text x="140" y="57" font-size="13">icpcg</text>',
      '<line x1="220" y1="52" x2="270" y2="52" stroke="#bb3e03" stroke-width="1.8" stroke-dasharray="6 4"/>',
      '<text x="280" y="57" font-size="13">paper_split_icpcg</text>',
  ]

  for idx, step in enumerate(args.steps):
    unsplit_path = Path(args.unsplit_dir) / f"{args.unsplit_name}_step_{step:06d}.vtk"
    split_path = Path(args.split_dir) / f"{args.split_name}_step_{step:06d}.vtk"
    x_u, y_u, field_u = read_phase_vtk(unsplit_path)
    x_s, y_s, field_s = read_phase_vtk(split_path)
    left = left0 + idx * (panel_w + gap)
    add_panel(lines, x_u, y_u, field_u, field_s, left, top, panel_w, panel_h, f"t = {step * args.dt:.3f}")

  lines.append("</svg>")
  with open(args.output, "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines))
  print(f"INTERFACE_COMPARE_OK output={args.output}")


if __name__ == "__main__":
  main()
