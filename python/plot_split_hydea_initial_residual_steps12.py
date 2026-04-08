#!/usr/bin/env python3

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/weiyy/phaseField/dinghang/ding_nsch_core_solver")
PRESSURE_DIR = (
    ROOT
    / "output"
    / "rising_bubble_ding2007_planar_split_hydea_alt_norm_10cg3ml_192_2"
    / "pressure_solver"
)
OUTPUT_DIR = ROOT / "output" / "hydea_initial_residual_spectrum_compare"

NX = 192
NY = 192


def read_vector(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as handle:
        count = int(handle.readline().strip())
        values = np.array([float(line.strip()) for line in handle if line.strip()], dtype=np.float64)
    if count != NX * NY or values.size != count:
        raise RuntimeError(f"unexpected vector size in {path}: header={count}, values={values.size}")
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    step1 = read_vector(PRESSURE_DIR / "rhs_step_000001.txt")
    step2 = read_vector(PRESSURE_DIR / "rhs_step_000002.txt")

    power1, radial1, k = compute_spectrum(step1)
    power2, radial2, _ = compute_spectrum(step2)

    power1_norm = power1 / np.max(power1)
    power2_norm = power2 / np.max(power2)
    radial1_norm = radial1 / np.max(radial1)
    radial2_norm = radial2 / np.max(radial2)
    cumulative1 = np.cumsum(radial1)
    cumulative2 = np.cumsum(radial2)
    cumulative1_norm = cumulative1 / np.maximum(cumulative1[-1], 1.0e-30)
    cumulative2_norm = cumulative2 / np.maximum(cumulative2[-1], 1.0e-30)

    csv_path = OUTPUT_DIR / "step1_step2_initial_residual_radial.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "wave_number",
                "step1_radial_power",
                "step2_radial_power",
                "step1_radial_power_normalized",
                "step2_radial_power_normalized",
                "step1_cumulative_energy_fraction",
                "step2_cumulative_energy_fraction",
            ]
        )
        for idx in range(k.size):
            writer.writerow(
                [
                    f"{k[idx]:.0f}",
                    f"{radial1[idx]:.17e}",
                    f"{radial2[idx]:.17e}",
                    f"{radial1_norm[idx]:.17e}",
                    f"{radial2_norm[idx]:.17e}",
                    f"{cumulative1_norm[idx]:.17e}",
                    f"{cumulative2_norm[idx]:.17e}",
                ]
            )

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    cmap = "magma"

    im0 = axes[0, 0].imshow(np.log10(power1_norm + 1.0e-16), origin="lower", cmap=cmap)
    axes[0, 0].set_title("Step 1 Initial Residual Spectrum\niter=0, log10 normalized power")
    axes[0, 0].set_xlabel("k_y index")
    axes[0, 0].set_ylabel("k_x index")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(np.log10(power2_norm + 1.0e-16), origin="lower", cmap=cmap)
    axes[0, 1].set_title("Step 2 Initial Residual Spectrum\niter=0, log10 normalized power")
    axes[0, 1].set_xlabel("k_y index")
    axes[0, 1].set_ylabel("k_x index")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[1, 0].plot(k[1:], radial1_norm[1:], label="Step 1", linewidth=2.0)
    axes[1, 0].plot(k[1:], radial2_norm[1:], label="Step 2", linewidth=2.0)
    axes[1, 0].set_title("Radially Averaged Power Spectrum\nNormalized by each step max")
    axes[1, 0].set_xlabel("Wave number")
    axes[1, 0].set_ylabel("Normalized power")
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(k[1:], cumulative1_norm[1:], label="Step 1", linewidth=2.0)
    axes[1, 1].plot(k[1:], cumulative2_norm[1:], label="Step 2", linewidth=2.0)
    axes[1, 1].set_title("Cumulative Spectral Energy Fraction")
    axes[1, 1].set_xlabel("Wave number cutoff K")
    axes[1, 1].set_ylabel("Cumulative energy fraction")
    axes[1, 1].set_xlim(left=1)
    axes[1, 1].set_ylim(0.0, 1.02)
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend(frameon=False)

    fig.suptitle("HyDEA Split Pressure Initial Residual Spectrum Comparison\nRising bubble, step 1 vs step 2")
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)

    svg_path = OUTPUT_DIR / "step1_step2_initial_residual_spectrum.svg"
    png_path = OUTPUT_DIR / "step1_step2_initial_residual_spectrum.png"
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
