#!/usr/bin/env python3

import argparse
import math
import pathlib
import re

import matplotlib.pyplot as plt


def load_deformation(path: pathlib.Path):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        rows.append(tuple(float(x) for x in parts[:6]))
    return rows


def load_segments(path: pathlib.Path):
    segments = []
    current_x = []
    current_y = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            if current_x:
                segments.append((current_x, current_y))
                current_x, current_y = [], []
            continue
        x_str, y_str = line.split()[:2]
        current_x.append(float(x_str))
        current_y.append(float(y_str))
    if current_x:
        segments.append((current_x, current_y))
    return segments


def plot_curve(rows, output_path: pathlib.Path):
    t = [row[0] for row in rows]
    d = [row[5] for row in rows]
    plt.figure(figsize=(6.5, 4.2))
    plt.plot(t, d, color="#0b5fff", lw=2.0)
    plt.xlabel(r"$\dot{\gamma} t$")
    plt.ylabel("Taylor deformation D")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def extract_time(path: pathlib.Path):
    match = re.search(r"_t(\d+\.\d+)\.dat$", path.name)
    if not match:
      raise ValueError(f"cannot parse time from {path}")
    return float(match.group(1))


def plot_shapes(interface_files, output_path: pathlib.Path):
    interface_files = sorted(interface_files, key=extract_time)
    times_to_plot = [0.0, 2.0, 4.0, 6.0, 7.0]
    selected = []
    for target in times_to_plot:
        best = min(interface_files, key=lambda p: abs(extract_time(p) - target))
        if best not in selected:
            selected.append(best)

    colors = ["#111111", "#1b7f3b", "#a35d00", "#b22222", "#0b5fff"]
    plt.figure(figsize=(5.5, 5.5))
    for color, path in zip(colors, selected):
        segments = load_segments(path)
        label = f"t={extract_time(path):.2f}"
        first = True
        for xs, ys in segments:
            plt.plot(xs, ys, color=color, lw=1.5, label=label if first else None)
            first = False
    plt.gca().set_aspect("equal")
    plt.xlim(-2.05, 2.05)
    plt.ylim(-2.05, 2.05)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(frameon=True, fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True)
    args = parser.parse_args()

    case_dir = pathlib.Path(args.case_dir)
    log_path = case_dir / "deformation.log"
    rows = load_deformation(log_path)
    plot_curve(rows, case_dir / "deformation_curve.svg")

    interface_files = list(case_dir.glob("interface_t*.dat"))
    plot_shapes(interface_files, case_dir / "interface_evolution.svg")


if __name__ == "__main__":
    main()
