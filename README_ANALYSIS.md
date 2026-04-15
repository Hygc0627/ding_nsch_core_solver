# Pressure Poisson PCG Analysis

## Build

```bash
cmake -S . -B build
cmake --build build -j4
```

## Run A Single Case

```bash
./build/ding_nsch testcases/70_static_droplet_pressure_analysis_g2_64.cfg
```

The solver writes analysis outputs under:

```text
output/<case_name>/pressure_analysis/
```

Important files:

- `summary.csv`: one row per analysis solve
- `*_pcg.csv`: one row per PCG iteration
- `*_fields/rhs.csv`: frozen RHS
- `*_fields/residual_iter_*.csv`: residual snapshots at configured key iterations
- `*_fields/residual_final.csv`: final residual
- `*_fields/interface_mask.csv`: interface-band mask
- `*_fields/rho_mid.csv`: frozen midpoint density
- `*_fields/phi.csv`: frozen order parameter `phi = 2c - 1`

## Recommended Static-Droplet Cases

- `testcases/68_static_droplet_pressure_analysis_g0_64.cfg`
- `testcases/69_static_droplet_pressure_analysis_g1_64.cfg`
- `testcases/70_static_droplet_pressure_analysis_g2_64.cfg`
- `testcases/71_static_droplet_pressure_analysis_g3_dr10_64.cfg`
- `testcases/72_static_droplet_pressure_analysis_g4_dr10_64.cfg`
- `testcases/73_static_droplet_pressure_analysis_g4_dr100_64.cfg`
- `testcases/74_static_droplet_pressure_analysis_g4_dr1000_64.cfg`

The same configs can be scaled to `128` or `256` by editing `nx`, `ny`, and, if needed, solver tolerances.

## Generate Plots And Report

```bash
python3 python/pressure_pcg_analysis_report.py \
  --case-dirs \
    output/static_droplet_pressure_analysis_g0_64 \
    output/static_droplet_pressure_analysis_g1_64 \
    output/static_droplet_pressure_analysis_g2_64 \
    output/static_droplet_pressure_analysis_g3_dr10_64 \
    output/static_droplet_pressure_analysis_g4_dr10_64 \
    output/static_droplet_pressure_analysis_g4_dr100_64 \
    output/static_droplet_pressure_analysis_g4_dr1000_64 \
  --source-mode frozen \
  --output-dir output/pressure_analysis_report
```

Generated outputs:

- `fig1_main_three_cases.{png,pdf}`
- `fig2_g0_g4.{png,pdf}`
- `fig3_qk.{png,pdf}`
- `fig4_interface_fraction.{png,pdf}`
- `fig5_density_scan.{png,pdf}`
- `fig6_rhs_spectrum.{png,pdf}`
- `fig7_residual_spectra.{png,pdf}`
- `fig8_spatial_fields.{png,pdf}`
- `spectra/*.csv`
- `report.md`

## Analysis Controls In Config

- `analysis_enabled = true|false`
- `analysis_mode = off|frozen|online|both`
- `analysis_trigger_step = <positive integer>`
- `analysis_case_group = G0|G1|G2|G3|G4`
- `analysis_initial_guess = zero`
- `analysis_nullspace_treatment = zero_mean_projection`
- `analysis_interface_band_multiplier = 2.0`
- `analysis_spectrum_iterations = 0,1,2,5,10,20,final`
- `surface_tension_multiplier = 0|1`

## Numerical Choices

- Pure Neumann pressure systems are handled by explicit zero-mean projection of the RHS, iterates, residuals, and solution vectors.
- The previous diagonal gauge shift was removed from the internal ICC/ILDLT PCG analysis path so that convergence curves reflect the singular Neumann operator on the mean-zero subspace.
- The interface band is defined from `phi = 2c - 1` and the distance-equivalent threshold `|phi| < tanh(alpha / (2 sqrt(2)))`, where `alpha = analysis_interface_band_multiplier`.
- The postprocessing spectrum uses 2D DCT-II with `norm="ortho"` from `scipy.fft.dctn`.

## Current Scope

- The new analysis toolchain is implemented for the internal `icpcg` and `ildlt_pcg` pressure solvers, including their split variants.
- `online` mode currently replays the same frozen linear system immediately after the requested step for those internal PCG solvers so the logged matrix, RHS, and initial guess remain identical to the in-run solve inputs.
