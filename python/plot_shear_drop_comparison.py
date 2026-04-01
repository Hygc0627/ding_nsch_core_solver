#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_phasefield(path: Path):
    times = []
    values = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            times.append(float(row["shear_time"]))
            values.append(float(row["taylor_deformation"]))
    return times, values


def load_basilisk(path: Path):
    times = []
    values = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            times.append(float(row["time"]))
            values.append(float(row["D"]))
    return times, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phasefield", required=True)
    parser.add_argument("--basilisk", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    phasefield_times, phasefield_values = load_phasefield(Path(args.phasefield))
    basilisk_times, basilisk_values = load_basilisk(Path(args.basilisk))

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(
        phasefield_times,
        phasefield_values,
        color="#0b5fff",
        lw=2.0,
        label="Phase field",
    )
    plt.plot(
        basilisk_times,
        basilisk_values,
        color="#d04a02",
        lw=2.0,
        label="Basilisk",
    )
    plt.xlabel(r"$\dot{\gamma} t$")
    plt.ylabel("Taylor deformation D")
    plt.xlim(left=0.0)
    plt.ylim(bottom=0.0)
    plt.grid(alpha=0.25)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(args.output)
    plt.close()


if __name__ == "__main__":
    main()
