#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path

from plot_zalesak_comparison import load_final_field, marching_squares, parse_config, zalesak_value


def load_metrics(path: Path):
    metrics = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics[row["cycles"]] = row
    return metrics


def write_summary_svg(output: Path, panels):
    cols = 2
    rows = 2
    panel_w = 430
    panel_h = 430
    plot_size = 300
    margin_x = 80
    margin_y = 72
    gutter_x = 40
    gutter_y = 40
    width = cols * panel_w + (cols - 1) * gutter_x
    height = rows * panel_h + (rows - 1) * gutter_y + 40

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width * 0.5}" y="32" text-anchor="middle" font-size="24" font-family="Arial">Zalesak Disk Advection: Cycle Summary</text>',
    ]

    def map_point(point, x0, y0):
        x, y = point
        sx = x0 + plot_size * x
        sy = y0 + plot_size * (1.0 - y)
        return sx, sy

    for idx, panel in enumerate(panels):
        col = idx % cols
        row = idx // cols
        px = col * (panel_w + gutter_x)
        py = 50 + row * (panel_h + gutter_y)
        plot_x = px + margin_x
        plot_y = py + margin_y

        lines.append(
            f'<rect x="{plot_x}" y="{plot_y}" width="{plot_size}" height="{plot_size}" fill="none" stroke="#333" stroke-width="1.4"/>'
        )
        lines.append(
            f'<text x="{px + panel_w * 0.5}" y="{py + 28}" text-anchor="middle" font-size="20" font-family="Arial">{panel["label"]}</text>'
        )

        for tick in range(6):
            value = tick / 5.0
            sx = plot_x + plot_size * value
            sy = plot_y + plot_size
            lines.append(f'<line x1="{sx:.2f}" y1="{sy}" x2="{sx:.2f}" y2="{sy + 6}" stroke="#333" stroke-width="1"/>')
            lines.append(
                f'<text x="{sx:.2f}" y="{sy + 24}" text-anchor="middle" font-size="12" font-family="Arial">{value:.1f}</text>'
            )
            sy2 = plot_y + plot_size * (1.0 - value)
            lines.append(
                f'<line x1="{plot_x - 6}" y1="{sy2:.2f}" x2="{plot_x}" y2="{sy2:.2f}" stroke="#333" stroke-width="1"/>'
            )
            lines.append(
                f'<text x="{plot_x - 10}" y="{sy2 + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial">{value:.1f}</text>'
            )

        lines.append(
            f'<text x="{plot_x + plot_size * 0.5}" y="{plot_y + plot_size + 44}" text-anchor="middle" font-size="14" font-family="Arial">x</text>'
        )
        lines.append(
            f'<text x="{plot_x - 48}" y="{plot_y + plot_size * 0.5}" text-anchor="middle" font-size="14" font-family="Arial" transform="rotate(-90 {plot_x - 48} {plot_y + plot_size * 0.5})">y</text>'
        )

        for p0, p1 in panel["initial_segments"]:
            x0, y0 = map_point(p0, plot_x, plot_y)
            x1, y1 = map_point(p1, plot_x, plot_y)
            lines.append(
                f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}" stroke="#111" stroke-width="1.5"/>'
            )
        for p0, p1 in panel["final_segments"]:
            x0, y0 = map_point(p0, plot_x, plot_y)
            x1, y1 = map_point(p1, plot_x, plot_y)
            lines.append(
                f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}" stroke="#c61b3a" stroke-width="1.1"/>'
            )

        box_w = 150
        box_h = 62
        box_x = plot_x + plot_size - box_w - 8
        box_y = plot_y + plot_size - box_h - 8
        text_x = box_x + 8
        text_y = box_y + 18
        lines.append(
            f'<rect x="{box_x}" y="{box_y}" width="{box_w}" height="{box_h}" fill="white" fill-opacity="0.88" stroke="#ccc"/>'
        )
        lines.append(
            f'<text x="{text_x}" y="{text_y}" font-size="12" font-family="Arial">L2_rms = {panel["l2"]:.3e}</text>'
        )
        lines.append(
            f'<text x="{text_x}" y="{text_y + 18}" font-size="12" font-family="Arial">Linf = {panel["linf"]:.3e}</text>'
        )
        lines.append(
            f'<text x="{text_x}" y="{text_y + 36}" font-size="12" font-family="Arial">c in [{panel["cmin"]:.3e}, {panel["cmax"]:.3e}]</text>'
        )

    legend_x = width - 220
    legend_y = 52
    lines.append(f'<rect x="{legend_x}" y="{legend_y}" width="170" height="54" fill="white" stroke="#bbb"/>')
    lines.append(
        f'<line x1="{legend_x + 12}" y1="{legend_y + 18}" x2="{legend_x + 42}" y2="{legend_y + 18}" stroke="#111" stroke-width="1.6"/>'
    )
    lines.append(
        f'<text x="{legend_x + 50}" y="{legend_y + 23}" font-size="13" font-family="Arial">initial c=0.5</text>'
    )
    lines.append(
        f'<line x1="{legend_x + 12}" y1="{legend_y + 37}" x2="{legend_x + 42}" y2="{legend_y + 37}" stroke="#c61b3a" stroke-width="1.2"/>'
    )
    lines.append(
        f'<text x="{legend_x + 50}" y="{legend_y + 42}" font-size="13" font-family="Arial">after N cycles</text>'
    )

    lines.append("</svg>")
    output.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Create a multi-cycle Zalesak contour summary figure.")
    parser.add_argument("--summary-csv", required=True, help="Cycle error summary CSV")
    parser.add_argument("--output", required=True, help="Output SVG path")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    metrics = load_metrics(Path(args.summary_csv))
    cycle_specs = [
        ("1", "1 cycle", root / "testcases/18_zalesak_disk_advection_128.cfg",
         root / "output/zalesak_disk_advection_128/final_cell_fields.csv"),
        ("2", "2 cycles", root / "testcases/19_zalesak_disk_advection_2cycles_128.cfg",
         root / "output/zalesak_disk_advection_2cycles_128/final_cell_fields.csv"),
        ("4", "4 cycles", root / "testcases/20_zalesak_disk_advection_4cycles_128.cfg",
         root / "output/zalesak_disk_advection_4cycles_128/final_cell_fields.csv"),
        ("8", "8 cycles", root / "testcases/21_zalesak_disk_advection_8cycles_128.cfg",
         root / "output/zalesak_disk_advection_8cycles_128/final_cell_fields.csv"),
    ]

    panels = []
    for cycle_key, label, cfg_path, final_csv_path in cycle_specs:
        cfg = parse_config(cfg_path)
        xs, ys, final_field = load_final_field(final_csv_path)
        initial_field = [[zalesak_value(xs[i], ys[j], cfg) for i in range(len(xs))] for j in range(len(ys))]
        panels.append(
            {
                "label": label,
                "initial_segments": marching_squares(xs, ys, initial_field, 0.5),
                "final_segments": marching_squares(xs, ys, final_field, 0.5),
                "l2": float(metrics[cycle_key]["l2_rms"]),
                "linf": float(metrics[cycle_key]["linf"]),
                "cmin": float(metrics[cycle_key]["c_min"]),
                "cmax": float(metrics[cycle_key]["c_max"]),
            }
        )

    write_summary_svg(Path(args.output), panels)


if __name__ == "__main__":
    main()
