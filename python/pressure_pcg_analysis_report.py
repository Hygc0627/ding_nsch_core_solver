#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dctn


def read_iteration_csv(path):
    rows = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = dict(row)
            for key in [
                "step",
                "time",
                "density_ratio",
                "sigma",
                "viscosity_ratio",
                "max_iter",
                "tolerance",
                "iter",
                "true_res_l2",
                "rel_true_res_l2",
                "true_res_linf",
                "precond_res",
                "alpha_k",
                "beta_k",
                "q_k",
                "interface_res_l2",
                "bulk_res_l2",
                "interface_res_fraction",
                "bulk_res_fraction",
            ]:
                parsed[key] = float(parsed[key]) if key not in {"step", "max_iter", "iter"} else int(float(parsed[key]))
            rows.append(parsed)
    return rows


def read_summary_csv(path):
    with path.open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    for row in rows:
        for key in [
            "step",
            "time",
            "density_ratio",
            "sigma",
            "viscosity_ratio",
            "max_iter",
            "tolerance",
            "rhs_volume_mean",
            "rhs_sum",
            "pressure_mean_before_fix",
            "pressure_mean_after_fix",
            "final_iter",
            "final_true_res_l2",
            "final_rel_true_res_l2",
        ]:
            row[key] = float(row[key]) if key not in {"step", "max_iter", "final_iter"} else int(float(row[key]))
    return rows


def load_matrix(path):
    return np.loadtxt(path, delimiter=",")


def radial_spectrum(field):
    coeff = dctn(field, type=2, norm="ortho")
    energy = np.abs(coeff) ** 2
    ny, nx = energy.shape
    ky, kx = np.indices((ny, nx))
    radius = np.sqrt(kx**2 + ky**2)
    bins = np.floor(radius + 0.5).astype(int)
    max_bin = int(bins.max())
    shell_energy = np.zeros(max_bin + 1)
    shell_count = np.zeros(max_bin + 1)
    for k in range(max_bin + 1):
        mask = bins == k
        shell_energy[k] = energy[mask].mean() if np.any(mask) else 0.0
        shell_count[k] = np.count_nonzero(mask)
    return coeff, energy, np.arange(max_bin + 1), shell_energy, shell_count


def export_spectrum(path, ks, energy):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["K", "E"])
        for k, e in zip(ks, energy):
            writer.writerow([int(k), float(e)])


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def pick_case(rows, group, sigma_positive=None, density_ratio=None):
    matches = [row for row in rows if row["case_group"] == group]
    if sigma_positive is not None:
      matches = [row for row in matches if (row["sigma"] > 0.0) == sigma_positive]
    if density_ratio is not None:
      matches = [row for row in matches if abs(row["density_ratio"] - density_ratio) < 1e-12]
    return matches[0] if matches else None


def label_for_summary(row):
    return f'{row["case_name"]} ({row["source_mode"]})'


def plot_semilogy(histories, y_key, output_path, title, ylabel):
    plt.figure(figsize=(7.0, 4.8))
    for label, rows in histories:
        xs = [row["iter"] for row in rows]
        ys = [max(row[y_key], 1.0e-30) for row in rows]
        plt.semilogy(xs, ys, marker="o", ms=3.0, lw=1.6, label=label)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", ls=":", lw=0.5)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path.with_suffix(".png"), dpi=180)
    plt.savefig(output_path.with_suffix(".pdf"))
    plt.close()


def plot_linear(histories, y_key, output_path, title, ylabel):
    plt.figure(figsize=(7.0, 4.8))
    for label, rows in histories:
        xs = [row["iter"] for row in rows]
        ys = [row[y_key] for row in rows]
        plt.plot(xs, ys, marker="o", ms=3.0, lw=1.6, label=label)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, ls=":", lw=0.5)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path.with_suffix(".png"), dpi=180)
    plt.savefig(output_path.with_suffix(".pdf"))
    plt.close()


def build_report(summary_rows, observations, output_path):
    lines = ["# Pressure PCG Analysis Report", ""]
    lines.append("## Observed")
    for item in observations["observed"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Inference")
    for item in observations["inferred"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Case Summary")
    lines.append("| case | group | source | final_iter | final_rel_true_res_l2 | sigma | density_ratio |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            f'| {row["case_name"]} | {row["case_group"]} | {row["source_mode"]} | {row["final_iter"]} | '
            f'{row["final_rel_true_res_l2"]:.3e} | {row["sigma"]:.3e} | {row["density_ratio"]:.1f} |'
        )
    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Postprocess pressure PCG analysis outputs.")
    parser.add_argument("--case-dirs", nargs="+", required=True, help="Case output directories that contain pressure_analysis/")
    parser.add_argument("--source-mode", default="frozen", choices=["frozen", "online"])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "spectra")

    summary_rows = []
    histories = {}
    field_dirs = {}
    for case_dir in args.case_dirs:
        analysis_dir = Path(case_dir) / "pressure_analysis"
        summary_path = analysis_dir / "summary.csv"
        if not summary_path.exists():
            continue
        selected_summary = [row for row in read_summary_csv(summary_path) if row["source_mode"] == args.source_mode]
        summary_rows.extend(selected_summary)
        for csv_path in sorted(analysis_dir.glob(f"*_{args.source_mode}_step_*_pcg.csv")):
            rows = read_iteration_csv(csv_path)
            if not rows:
                continue
            histories[rows[0]["case_name"]] = rows
            field_dirs[rows[0]["case_name"]] = analysis_dir / csv_path.stem.replace("_pcg", "_fields")

    summary_rows.sort(key=lambda row: row["case_name"])
    if not summary_rows:
        raise SystemExit("No analysis outputs found for the requested source mode.")

    g0 = pick_case(summary_rows, "G0")
    g2 = pick_case(summary_rows, "G2", sigma_positive=True)
    g4_sigma = [row for row in summary_rows if row["case_group"] == "G4" and row["sigma"] > 0.0]
    g4_sigma.sort(key=lambda row: row["density_ratio"])
    g4 = g4_sigma[0] if g4_sigma else None

    fig1_cases = [row for row in [g0, g2, g4] if row is not None]
    plot_semilogy(
        [(label_for_summary(row), histories[row["case_name"]]) for row in fig1_cases],
        "rel_true_res_l2",
        output_dir / "fig1_main_three_cases",
        "Main Cases: Relative True Residual",
        "rel_true_res_l2",
    )

    ordered_groups = []
    for group in ["G0", "G1", "G2", "G3", "G4"]:
        group_rows = [row for row in summary_rows if row["case_group"] == group]
        group_rows.sort(key=lambda row: (row["density_ratio"], row["sigma"]))
        ordered_groups.extend(group_rows[:1] if group != "G4" else group_rows[:1])
    plot_semilogy(
        [(label_for_summary(row), histories[row["case_name"]]) for row in ordered_groups if row["case_name"] in histories],
        "rel_true_res_l2",
        output_dir / "fig2_g0_g4",
        "G0-G4 Relative True Residual",
        "rel_true_res_l2",
    )

    plot_linear(
        [(label_for_summary(row), histories[row["case_name"]]) for row in fig1_cases],
        "q_k",
        output_dir / "fig3_qk",
        "Residual Reduction Ratio q_k",
        "q_k",
    )

    plot_linear(
        [(label_for_summary(row), histories[row["case_name"]]) for row in fig1_cases if row["case_name"] in histories],
        "interface_res_fraction",
        output_dir / "fig4_interface_fraction",
        "Interface Residual Fraction",
        "interface_res_fraction",
    )

    density_scan = []
    for density in [1.0, 10.0, 100.0, 1000.0]:
        candidate = None
        for row in summary_rows:
            if row["sigma"] > 0.0 and abs(row["density_ratio"] - density) < 1.0e-12:
                candidate = row
                break
        if candidate is not None:
            density_scan.append(candidate)
    plot_semilogy(
        [(f"dr={int(row['density_ratio'])}", histories[row["case_name"]]) for row in density_scan],
        "rel_true_res_l2",
        output_dir / "fig5_density_scan",
        "Density-Ratio Scan",
        "rel_true_res_l2",
    )

    representative = max(summary_rows, key=lambda row: (row["final_iter"], row["density_ratio"], row["sigma"]))
    rep_field_dir = field_dirs[representative["case_name"]]
    rhs = load_matrix(rep_field_dir / "rhs.csv")
    _, _, k_rhs, e_rhs, _ = radial_spectrum(rhs)
    export_spectrum(output_dir / "spectra" / f'{representative["case_name"]}_rhs_spectrum.csv', k_rhs, e_rhs)
    plt.figure(figsize=(7.0, 4.8))
    plt.semilogy(k_rhs, np.maximum(e_rhs, 1.0e-30), lw=1.8)
    plt.xlabel("K")
    plt.ylabel("E_b(K)")
    plt.title(f'RHS Spectrum: {representative["case_name"]}')
    plt.grid(True, which="both", ls=":", lw=0.5)
    plt.tight_layout()
    plt.savefig((output_dir / "fig6_rhs_spectrum").with_suffix(".png"), dpi=180)
    plt.savefig((output_dir / "fig6_rhs_spectrum").with_suffix(".pdf"))
    plt.close()

    plt.figure(figsize=(7.0, 4.8))
    rep_history = histories[representative["case_name"]]
    requested_iters = [0, 1, 2, 5, 10, 20, rep_history[-1]["iter"]]
    for iteration in requested_iters:
        field_path = rep_field_dir / ("residual_final.csv" if iteration == rep_history[-1]["iter"] else f"residual_iter_{iteration:06d}.csv")
        if not field_path.exists():
            continue
        residual = load_matrix(field_path)
        _, _, ks, energy, _ = radial_spectrum(residual)
        export_spectrum(output_dir / "spectra" / f'{representative["case_name"]}_residual_iter_{iteration:06d}.csv', ks, energy)
        plt.semilogy(ks, np.maximum(energy, 1.0e-30), lw=1.5, label=f"k={iteration}")
    plt.xlabel("K")
    plt.ylabel("E_r(K)")
    plt.title(f'Residual Spectrum: {representative["case_name"]}')
    plt.grid(True, which="both", ls=":", lw=0.5)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig((output_dir / "fig7_residual_spectra").with_suffix(".png"), dpi=180)
    plt.savefig((output_dir / "fig7_residual_spectra").with_suffix(".pdf"))
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0))
    fields = [
        ("RHS", rhs),
        ("Interface Mask", load_matrix(rep_field_dir / "interface_mask.csv")),
        ("Residual Final", load_matrix(rep_field_dir / "residual_final.csv")),
        ("rho_mid", load_matrix(rep_field_dir / "rho_mid.csv")),
    ]
    for ax, (title, field) in zip(axes.ravel(), fields):
        image = ax.imshow(field, origin="lower", cmap="coolwarm")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.75)
    fig.suptitle(f"Spatial Fields: {representative['case_name']}")
    fig.tight_layout()
    fig.savefig((output_dir / "fig8_spatial_fields").with_suffix(".png"), dpi=180)
    fig.savefig((output_dir / "fig8_spatial_fields").with_suffix(".pdf"))
    plt.close(fig)

    hardest = max(summary_rows, key=lambda row: row["final_iter"])
    observed = [
        f'The slowest case in this batch is `{hardest["case_name"]}` with final_iter={hardest["final_iter"]}.',
        f'The representative high-contrast case used for spectra is `{representative["case_name"]}`.',
    ]
    inferred = []

    g1 = pick_case(summary_rows, "G1", sigma_positive=False)
    if g1 and g2:
        observed.append(
            f'Surface tension changes the equal-density case from final_iter={g1["final_iter"]} to final_iter={g2["final_iter"]}.'
        )
        inferred.append("This suggests surface tension modifies the forcing spectrum that feeds the pressure solve.")
    g3 = pick_case(summary_rows, "G3", sigma_positive=False)
    if g2 and g4:
        observed.append(
            f'Adding variable density to the sigma-on case changes final_iter from {g2["final_iter"]} to {g4["final_iter"]}.'
        )
        inferred.append("This is consistent with variable coefficients making the Poisson operator harder for PCG.")

    rep_interface_mean = np.mean([row["interface_res_fraction"] for row in rep_history])
    observed.append(f'The representative case has mean interface_res_fraction={rep_interface_mean:.3f}.')
    if rep_interface_mean > 0.5:
        inferred.append("Residual energy is interface-dominated in the representative case.")
    else:
        inferred.append("Residual energy is not confined to the interface band alone.")

    build_report(summary_rows, {"observed": observed, "inferred": inferred}, output_dir / "report.md")


if __name__ == "__main__":
    main()
