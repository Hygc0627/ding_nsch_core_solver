#!/usr/bin/env python3

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/weiyy/phaseField/dinghang/ding_nsch_core_solver")
OUTPUT_ROOT = ROOT / "output"

UNSPLIT_DIR = OUTPUT_ROOT / "rising_bubble_ding2007_planar_unsplit_petsc_192_50" / "pressure_solver"
SPLIT_DIR = OUTPUT_ROOT / "rising_bubble_ding2007_planar_split_petsc_192_50" / "pressure_solver"
FIG_DIR = OUTPUT_ROOT / "rising_bubble_rhs_spectrum_compare"

NX = 192
NY = 192


def read_rhs(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as handle:
        count = int(handle.readline().strip())
        values = np.array([float(line.strip()) for line in handle if line.strip()], dtype=np.float64)
    if count != NX * NY or values.size != count:
        raise RuntimeError(f"unexpected rhs size in {path}: header={count}, values={values.size}")
    return values.reshape(NX, NY)


def compute_spectrum(field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = field - np.mean(field)
    fft = np.fft.fftshift(np.fft.fft2(centered))
    power = np.abs(fft) ** 2

    ky = np.fft.fftshift(np.fft.fftfreq(NY)) * NY
    kx = np.fft.fftshift(np.fft.fftfreq(NX)) * NX
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    radius = np.sqrt(kx_grid**2 + ky_grid**2)
    bins = np.floor(radius).astype(int)
    max_bin = int(bins.max())

    radial_sum = np.bincount(bins.ravel(), weights=power.ravel(), minlength=max_bin + 1)
    radial_count = np.bincount(bins.ravel(), minlength=max_bin + 1)
    radial = radial_sum / np.maximum(radial_count, 1)
    wave_number = np.arange(max_bin + 1, dtype=np.float64)
    return power, radial, wave_number


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    unsplit_rhs = read_rhs(UNSPLIT_DIR / "rhs.txt")
    split_rhs = read_rhs(SPLIT_DIR / "rhs.txt")

    unsplit_power, unsplit_radial, k = compute_spectrum(unsplit_rhs)
    split_power, split_radial, _ = compute_spectrum(split_rhs)

    unsplit_power_norm = unsplit_power / np.max(unsplit_power)
    split_power_norm = split_power / np.max(split_power)
    unsplit_radial_norm = unsplit_radial / np.max(unsplit_radial)
    split_radial_norm = split_radial / np.max(split_radial)
    unsplit_cumulative = np.cumsum(unsplit_radial)
    split_cumulative = np.cumsum(split_radial)
    unsplit_cumulative_norm = unsplit_cumulative / np.maximum(unsplit_cumulative[-1], 1.0e-30)
    split_cumulative_norm = split_cumulative / np.maximum(split_cumulative[-1], 1.0e-30)

    csv_path = FIG_DIR / "rhs_spectrum_radial_step50.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "wave_number",
                "unsplit_radial_power",
                "split_radial_power",
                "unsplit_radial_power_normalized",
                "split_radial_power_normalized",
                "unsplit_cumulative_energy_fraction",
                "split_cumulative_energy_fraction",
            ]
        )
        for idx in range(k.size):
            writer.writerow(
                [
                    f"{k[idx]:.0f}",
                    f"{unsplit_radial[idx]:.17e}",
                    f"{split_radial[idx]:.17e}",
                    f"{unsplit_radial_norm[idx]:.17e}",
                    f"{split_radial_norm[idx]:.17e}",
                    f"{unsplit_cumulative_norm[idx]:.17e}",
                    f"{split_cumulative_norm[idx]:.17e}",
                ]
            )

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    cmap = "magma"

    im0 = axes[0, 0].imshow(np.log10(unsplit_power_norm + 1.0e-16), origin="lower", cmap=cmap)
    axes[0, 0].set_title("Unsplit RHS Spectrum\nStep 50, log10 normalized power")
    axes[0, 0].set_xlabel("k_y index")
    axes[0, 0].set_ylabel("k_x index")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(np.log10(split_power_norm + 1.0e-16), origin="lower", cmap=cmap)
    axes[0, 1].set_title("Split RHS Spectrum\nStep 50, log10 normalized power")
    axes[0, 1].set_xlabel("k_y index")
    axes[0, 1].set_ylabel("k_x index")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[1, 0].semilogy(k[1:], unsplit_radial[1:], label="Unsplit", linewidth=2.0)
    axes[1, 0].semilogy(k[1:], split_radial[1:], label="Split", linewidth=2.0)
    axes[1, 0].set_title("Radially Averaged Power Spectrum\nRaw magnitude")
    axes[1, 0].set_xlabel("Wave number")
    axes[1, 0].set_ylabel("Power")
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(k[1:], unsplit_radial_norm[1:], label="Unsplit", linewidth=2.0)
    axes[1, 1].plot(k[1:], split_radial_norm[1:], label="Split", linewidth=2.0)
    axes[1, 1].set_title("Radially Averaged Power Spectrum\nNormalized by each case max")
    axes[1, 1].set_xlabel("Wave number")
    axes[1, 1].set_ylabel("Normalized power")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend(frameon=False)

    fig.suptitle("Pressure Poisson RHS Spectrum Comparison at Step 50\nVariable-density rising bubble")
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)

    svg_path = FIG_DIR / "rhs_spectrum_compare_step50.svg"
    png_path = FIG_DIR / "rhs_spectrum_compare_step50.png"
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    cumulative_fig, cumulative_ax = plt.subplots(figsize=(8.5, 5.5))
    cumulative_ax.plot(k[1:], unsplit_cumulative_norm[1:], label="Unsplit", linewidth=2.0)
    cumulative_ax.plot(k[1:], split_cumulative_norm[1:], label="Split", linewidth=2.0)
    cumulative_ax.set_title("Cumulative Spectral Energy Fraction at Step 50")
    cumulative_ax.set_xlabel("Wave number cutoff K")
    cumulative_ax.set_ylabel("Cumulative energy fraction")
    cumulative_ax.set_xlim(left=1)
    cumulative_ax.set_ylim(0.0, 1.02)
    cumulative_ax.grid(True, alpha=0.25)
    cumulative_ax.legend(frameon=False)
    cumulative_fig.tight_layout()

    cumulative_svg_path = FIG_DIR / "rhs_spectrum_cumulative_step50.svg"
    cumulative_png_path = FIG_DIR / "rhs_spectrum_cumulative_step50.png"
    cumulative_fig.savefig(cumulative_svg_path, bbox_inches="tight")
    cumulative_fig.savefig(cumulative_png_path, dpi=200, bbox_inches="tight")
    plt.close(cumulative_fig)


if __name__ == "__main__":
    main()
