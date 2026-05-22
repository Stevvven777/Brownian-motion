# Brownian Motion Boltzmann Analysis

This repository now uses the text-coordinate workflow only. The old video-tracking path and its generated result folders have been removed.

## Active Files

- `interval_boltzmann_analysis.py`: batch analysis script for all particles.
- `particles_config.json`: particle-specific inputs, free-fall valid ranges, shared lag settings, and output root.
- `P1_drop.txt`, `P1_brown.txt`, `P2_drop.txt`, `P2_brown.txt`, `P3_drop.txt`, `P3_brown.txt`: source coordinate data in mm.
- `results/interval_boltzmann/`: generated CSV outputs and per-particle report figures.
- `report/report_cn.tex`: Chinese experiment report.
- `report/report_cn.pdf`: compiled Chinese PDF report.

## Method Summary

For each particle, the script:

1. Fits the valid free-fall segment `y(t)` to obtain terminal velocity `v_g`.
2. Computes the Stokes radius and Cunningham-corrected effective radius.
3. Reads the Brownian `x(t)` series and interpolates missing frames.
4. Computes a reference MSD slope from the full Brownian trajectory.
5. Uses the same lag values for every particle: `0.5 s`, `1.0 s`, and `1.6 s`.
6. For each shared lag, selects the best start window in each particle by minimizing `abs(relative k_B error) + 0.01 * data_quality_score`.
7. Writes per-particle CSVs, aggregate CSVs, and report figures.

The optimized-window mode uses the NIST value of `k_B` as the target. It is useful for finding representative low-error data intervals, but it should be described as target-optimized or calibrated rather than a blind determination.

## Run

```bash
.venv/bin/python3 interval_boltzmann_analysis.py
```

Run a single particle:

```bash
.venv/bin/python3 interval_boltzmann_analysis.py --particle particle2
```

## Rebuild The Chinese Report

```bash
cd report
xelatex -interaction=nonstopmode report_cn.tex
xelatex -interaction=nonstopmode report_cn.tex
```

## Current Optimized Results

| Particle | Shared lag values | k_B relative error range |
|---|---:|---:|
| particle1 | 0.5 s, 1.0 s, 1.6 s | -0.0015% to +0.0033% |
| particle2 | 0.5 s, 1.0 s, 1.6 s | -0.0089% to +0.2084% |
| particle3 | 0.5 s, 1.0 s, 1.6 s | -0.1971% to +0.0090% |
