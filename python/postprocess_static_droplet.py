#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_cfg(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        cfg[key] = value
    return cfg


def analytic_delta_profile(x: np.ndarray, x_interface: float, cn: float) -> np.ndarray:
    # For c(x) = 0.5 - 0.5 * tanh((x - x_i) / (2*sqrt(2)*Cn))
    # the regularized interfacial delta is -dc/dx.
    width = 2.0 * math.sqrt(2.0) * cn
    z = (x - x_interface) / width
    return 1.0 / (4.0 * math.sqrt(2.0) * cn) * (1.0 / np.cosh(z) ** 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess a static-droplet result directory.")
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--cfg", required=True, type=Path)
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    cfg = parse_cfg(args.cfg.resolve())

    df = pd.read_csv(results_dir / "final_cell_fields.csv")
    hist = pd.read_csv(results_dir / "history.csv")

    nx = int(cfg["nx"])
    ny = int(cfg["ny"])
    re = float(cfg["re"])
    ca = float(cfg["ca"])
    cn = float(cfg["cn"])
    sigma_mult = float(cfg.get("surface_tension_multiplier", "1.0"))
    cx = float(cfg["interface_center_x"])
    cy = float(cfg["interface_center_y"])
    radius = float(cfg["interface_radius"])

    x_vals = np.sort(df["x"].unique())
    y_vals = np.sort(df["y"].unique())
    dx = float(np.mean(np.diff(x_vals)))
    dy = float(np.mean(np.diff(y_vals)))

    c = df.pivot(index="j", columns="i", values="c").sort_index().to_numpy()
    pressure = df.pivot(index="j", columns="i", values="pressure").sort_index().to_numpy()
    mu = df.pivot(index="j", columns="i", values="chemical_potential").sort_index().to_numpy()
    u = df.pivot(index="j", columns="i", values="u").sort_index().to_numpy()
    v = df.pivot(index="j", columns="i", values="v").sort_index().to_numpy()

    dc_dy, dc_dx = np.gradient(c, dy, dx)
    scale = sigma_mult / (re * ca)
    fx = scale * mu * dc_dx
    fy = scale * mu * dc_dy
    fmag = np.hypot(fx, fy)
    speed = np.hypot(u, v)

    x_grid = x_vals[None, :].repeat(ny, axis=0)
    y_grid = y_vals[:, None].repeat(nx, axis=1)
    r = np.hypot(x_grid - cx, y_grid - cy)

    interface_half_width = 2.0 * math.sqrt(2.0) * cn
    inside_mask = (c > 0.9) & (r < radius)
    outside_mask = (c < 0.1) & (r > radius)
    interface_band = (c > 0.05) & (c < 0.95)

    p_inside = float(np.mean(pressure[inside_mask]))
    p_outside = float(np.mean(pressure[outside_mask]))
    dp_measured = p_inside - p_outside
    dp_theory = scale / radius

    max_speed = float(speed.max())
    max_speed_idx = np.unravel_index(np.argmax(speed), speed.shape)
    max_speed_x = float(x_grid[max_speed_idx])
    max_speed_y = float(y_grid[max_speed_idx])
    interface_max_speed = float(speed[interface_band].max())

    j_center = int(np.argmin(np.abs(y_vals - cy)))
    x_line = x_vals
    pressure_line = pressure[j_center, :]
    fx_line = fx[j_center, :]
    speed_line = speed[j_center, :]

    x_interface = cx + radius
    x_window_mask = np.abs(x_line - x_interface) <= max(0.1, 8.0 * cn)
    x_window = x_line[x_window_mask]
    f_num_window = -fx_line[x_window_mask]
    f_theory_window = dp_theory * analytic_delta_profile(x_window, x_interface, cn)

    out_dir = results_dir / "postprocess_static_droplet"
    out_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(6.5, 5.6))
    extent = [x_vals[0] - 0.5 * dx, x_vals[-1] + 0.5 * dx, y_vals[0] - 0.5 * dy, y_vals[-1] + 0.5 * dy]
    im = plt.imshow(
        fmag,
        origin="lower",
        extent=extent,
        cmap="magma",
        interpolation="nearest",
    )
    plt.colorbar(im, label=r"$|f_\sigma|$")
    plt.contour(x_vals, y_vals, c, levels=[0.5], colors="cyan", linewidths=0.8)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Static Droplet Surface-Tension Force Magnitude")
    plt.tight_layout()
    plt.savefig(out_dir / "surface_tension_force_magnitude.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7.0, 4.8))
    plt.plot(x_window, f_num_window, label="Numerical $-f_{\\sigma,x}$ at $y=y_c$", linewidth=2.0)
    plt.plot(x_window, f_theory_window, "--", label="Theory $\\Delta p_{Laplace}\\,\\delta_\\epsilon$", linewidth=2.0)
    plt.xlabel("x")
    plt.ylabel("Force Density")
    plt.title("Right-Interface Surface-Tension Distribution vs Theory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "surface_tension_distribution_vs_theory.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7.0, 4.8))
    plt.plot(x_line, pressure_line, label="Pressure on centerline", linewidth=2.0)
    plt.axhline(p_inside, color="tab:green", linestyle="--", label=f"Inside mean = {p_inside:.6e}")
    plt.axhline(p_outside, color="tab:red", linestyle="--", label=f"Outside mean = {p_outside:.6e}")
    plt.axhline(p_outside + dp_theory, color="black", linestyle=":", label=f"Theory inside = {p_outside + dp_theory:.6e}")
    plt.xlabel("x")
    plt.ylabel("Pressure")
    plt.title("Pressure Jump vs Laplace Theory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pressure_jump_vs_theory.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6.5, 5.6))
    im = plt.imshow(speed, origin="lower", extent=extent, cmap="viridis", interpolation="nearest")
    plt.colorbar(im, label=r"$|\mathbf{u}|$")
    skip = 4
    plt.quiver(
        x_grid[::skip, ::skip],
        y_grid[::skip, ::skip],
        u[::skip, ::skip],
        v[::skip, ::skip],
        color="white",
        scale=1.5e-5,
        width=0.002,
    )
    plt.contour(x_vals, y_vals, c, levels=[0.5], colors="orange", linewidths=0.8)
    plt.scatter([max_speed_x], [max_speed_y], color="red", s=30, label=f"max |u| = {max_speed:.3e}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Spurious Current Magnitude and Direction")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / "spurious_current_magnitude.png", dpi=200)
    plt.close()

    report_lines = [
        f"results_dir,{results_dir}",
        f"cfg,{args.cfg.resolve()}",
        f"nx,{nx}",
        f"ny,{ny}",
        f"dx,{dx:.12e}",
        f"dy,{dy:.12e}",
        f"radius,{radius:.12e}",
        f"center_x,{cx:.12e}",
        f"center_y,{cy:.12e}",
        f"surface_force_scale,1/(Re*Ca)={scale:.12e}",
        f"pressure_inside_mean,{p_inside:.12e}",
        f"pressure_outside_mean,{p_outside:.12e}",
        f"pressure_jump_measured,{dp_measured:.12e}",
        f"pressure_jump_theory,{dp_theory:.12e}",
        f"pressure_jump_relative_error,{(dp_measured - dp_theory) / dp_theory:.12e}",
        f"max_velocity_summary_csv,{hist['max_velocity'].iloc[-1]:.12e}",
        f"max_velocity_field,{max_speed:.12e}",
        f"max_velocity_x,{max_speed_x:.12e}",
        f"max_velocity_y,{max_speed_y:.12e}",
        f"interface_band_max_velocity,{interface_max_speed:.12e}",
        f"max_surface_force,{fmag.max():.12e}",
        f"line_integral_minus_fx_right_interface,{np.trapz(f_num_window, x_window):.12e}",
        f"line_integral_theory_right_interface,{np.trapz(f_theory_window, x_window):.12e}",
    ]
    (out_dir / "report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
