#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path


def parse_config(path: Path) -> dict:
    cfg = {}
    with path.open() as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            cfg[key.strip()] = value.strip()
    return cfg


def signed_distance_box(x: float, y: float, cx: float, cy: float, hx: float, hy: float) -> float:
    qx = abs(x - cx) - hx
    qy = abs(y - cy) - hy
    ox = max(qx, 0.0)
    oy = max(qy, 0.0)
    return math.hypot(ox, oy) + min(max(qx, qy), 0.0)


def zalesak_value(x: float, y: float, cfg: dict) -> float:
    cx = float(cfg["interface_center_x"])
    cy = float(cfg["interface_center_y"])
    radius = float(cfg["interface_radius"])
    slot_width = float(cfg["zalesak_slot_width"])
    slot_depth = float(cfg["zalesak_slot_depth"])
    cn = float(cfg["cn"])
    slot_half_width = 0.5 * slot_width
    slot_top = cy + radius
    slot_bottom = slot_top - slot_depth
    slot_cy = 0.5 * (slot_top + slot_bottom)
    slot_half_height = 0.5 * (slot_top - slot_bottom)
    smoothing = max(2.0 * math.sqrt(2.0) * cn, 1.0e-12)

    disk_sd = math.hypot(x - cx, y - cy) - radius
    slot_sd = signed_distance_box(x, y, cx, slot_cy, slot_half_width, slot_half_height)
    shape_sd = max(disk_sd, -slot_sd)
    return 0.5 - 0.5 * math.tanh(shape_sd / smoothing)


def load_final_field(path: Path):
    xs = {}
    ys = {}
    values = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            i = int(row["i"])
            j = int(row["j"])
            xs[i] = float(row["x"])
            ys[j] = float(row["y"])
            values[(i, j)] = float(row["c"])

    nx = max(xs) + 1
    ny = max(ys) + 1
    x_list = [xs[i] for i in range(nx)]
    y_list = [ys[j] for j in range(ny)]
    field = [[values[(i, j)] for i in range(nx)] for j in range(ny)]
    return x_list, y_list, field


def interpolate(p0, p1, v0, v1, level):
    dv = v1 - v0
    if abs(dv) < 1.0e-14:
        return ((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5)
    t = (level - v0) / dv
    t = max(0.0, min(1.0, t))
    return (p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]))


def marching_squares(xs, ys, field, level):
    segments = []
    nx = len(xs)
    ny = len(ys)
    for j in range(ny - 1):
        for i in range(nx - 1):
            p0 = (xs[i], ys[j])
            p1 = (xs[i + 1], ys[j])
            p2 = (xs[i + 1], ys[j + 1])
            p3 = (xs[i], ys[j + 1])
            v0 = field[j][i]
            v1 = field[j][i + 1]
            v2 = field[j + 1][i + 1]
            v3 = field[j + 1][i]
            idx = 0
            if v0 >= level:
                idx |= 1
            if v1 >= level:
                idx |= 2
            if v2 >= level:
                idx |= 4
            if v3 >= level:
                idx |= 8
            if idx == 0 or idx == 15:
                continue

            edges = {
                0: interpolate(p0, p1, v0, v1, level),
                1: interpolate(p1, p2, v1, v2, level),
                2: interpolate(p2, p3, v2, v3, level),
                3: interpolate(p3, p0, v3, v0, level),
            }

            center = 0.25 * (v0 + v1 + v2 + v3)
            table = {
                1: [(3, 0)],
                2: [(0, 1)],
                3: [(3, 1)],
                4: [(1, 2)],
                5: [(3, 0), (1, 2)] if center >= level else [(3, 2), (0, 1)],
                6: [(0, 2)],
                7: [(3, 2)],
                8: [(2, 3)],
                9: [(0, 2)],
                10: [(0, 1), (2, 3)] if center >= level else [(3, 0), (1, 2)],
                11: [(1, 2)],
                12: [(3, 1)],
                13: [(0, 1)],
                14: [(3, 0)],
            }
            for e0, e1 in table[idx]:
                segments.append((edges[e0], edges[e1]))
    return segments


def write_svg(output: Path, initial_segments, final_segments, domain_min=0.0, domain_max=1.0):
    width = 760
    height = 760
    margin = 70
    plot = width - 2 * margin

    def map_point(point):
        x, y = point
        sx = margin + plot * (x - domain_min) / (domain_max - domain_min)
        sy = height - margin - plot * (y - domain_min) / (domain_max - domain_min)
        return sx, sy

    lines = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    lines.append('<rect width="100%" height="100%" fill="white"/>')
    lines.append(
        f'<rect x="{margin}" y="{margin}" width="{plot}" height="{plot}" fill="none" stroke="#333" stroke-width="1.5"/>'
    )
    lines.append(
        f'<text x="{width * 0.5}" y="36" text-anchor="middle" font-size="22" font-family="Arial">Zalesak Disk: Initial vs Final c=0.5 Contour</text>'
    )

    for x_tick in range(6):
        value = domain_min + (domain_max - domain_min) * x_tick / 5.0
        sx = margin + plot * x_tick / 5.0
        sy = height - margin
        lines.append(f'<line x1="{sx:.2f}" y1="{sy}" x2="{sx:.2f}" y2="{sy + 8}" stroke="#333" stroke-width="1"/>')
        lines.append(
            f'<text x="{sx:.2f}" y="{sy + 28}" text-anchor="middle" font-size="14" font-family="Arial">{value:.1f}</text>'
        )
        sy2 = height - margin - plot * x_tick / 5.0
        lines.append(
            f'<line x1="{margin - 8}" y1="{sy2:.2f}" x2="{margin}" y2="{sy2:.2f}" stroke="#333" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{margin - 14}" y="{sy2 + 5:.2f}" text-anchor="end" font-size="14" font-family="Arial">{value:.1f}</text>'
        )

    lines.append(
        f'<text x="{width * 0.5}" y="{height - 18}" text-anchor="middle" font-size="16" font-family="Arial">x</text>'
    )
    lines.append(
        f'<text x="20" y="{height * 0.5}" text-anchor="middle" font-size="16" font-family="Arial" transform="rotate(-90 20 {height * 0.5})">y</text>'
    )

    for p0, p1 in initial_segments:
        x0, y0 = map_point(p0)
        x1, y1 = map_point(p1)
        lines.append(
            f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}" stroke="#111" stroke-width="1.6"/>'
        )
    for p0, p1 in final_segments:
        x0, y0 = map_point(p0)
        x1, y1 = map_point(p1)
        lines.append(
            f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}" stroke="#c61b3a" stroke-width="1.2"/>'
        )

    legend_x = width - 235
    legend_y = 80
    lines.append(f'<rect x="{legend_x}" y="{legend_y}" width="165" height="56" fill="white" stroke="#bbb"/>')
    lines.append(
        f'<line x1="{legend_x + 12}" y1="{legend_y + 18}" x2="{legend_x + 42}" y2="{legend_y + 18}" stroke="#111" stroke-width="1.8"/>'
    )
    lines.append(
        f'<text x="{legend_x + 50}" y="{legend_y + 23}" font-size="14" font-family="Arial">initial</text>'
    )
    lines.append(
        f'<line x1="{legend_x + 12}" y1="{legend_y + 38}" x2="{legend_x + 42}" y2="{legend_y + 38}" stroke="#c61b3a" stroke-width="1.4"/>'
    )
    lines.append(
        f'<text x="{legend_x + 50}" y="{legend_y + 43}" font-size="14" font-family="Arial">after one turn</text>'
    )

    lines.append("</svg>")
    output.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Plot initial/final Zalesak disk contour comparison.")
    parser.add_argument("--config", required=True, help="Path to the Zalesak config file")
    parser.add_argument("--final-csv", required=True, help="Path to final_cell_fields.csv")
    parser.add_argument("--output", required=True, help="Output SVG path")
    args = parser.parse_args()

    config_path = Path(args.config)
    final_csv_path = Path(args.final_csv)
    output_path = Path(args.output)

    cfg = parse_config(config_path)
    xs, ys, final_field = load_final_field(final_csv_path)
    initial_field = [[zalesak_value(xs[i], ys[j], cfg) for i in range(len(xs))] for j in range(len(ys))]

    initial_segments = marching_squares(xs, ys, initial_field, 0.5)
    final_segments = marching_squares(xs, ys, final_field, 0.5)
    write_svg(output_path, initial_segments, final_segments, 0.0, float(cfg.get("lx", "1.0")))


if __name__ == "__main__":
    main()
