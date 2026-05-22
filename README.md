# Millikan Brownian kB Analyzer

This workspace contains a Python script for estimating Boltzmann's constant from Millikan oil-drop videos. It follows the analysis flow in *Exploring Brownian Motion with a Millikan Oil Drop Apparatus* and uses the standard PASCO/Millikan Cunningham-corrected radius formula referenced by that article metadata.

The local PDF could not be machine-extracted with the available tools, so the implementation keeps the slip constant configurable. The default radius model is:

```text
r = sqrt((b / (2 p))^2 + 9 eta v_g / (2 (rho_oil - rho_air) g)) - b / (2 p)
C = 1 + b / (p r)
eta_slip = eta_air / C
k_B = 6 pi eta_slip r D / T
```

Default `b` is `8.20e-3 Pa*m`, matching the common PASCO/Millikan form. Override it with `--cunningham-b-pa-m` if your paper copy gives a different value.

## Video requirements

### Format and codec

Any container and codec that OpenCV can open: `.mp4` (H.264/H.265), `.avi`, `.mov`, `.mkv`. The FPS value must be stored in the file header; if it is missing or wrong, pass `--fps`.

### Video A — free fall (no electric field)

| Property | Requirement |
|---|---|
| Content | Drops fall under gravity only; electric field off |
| Duration | ≥ 3 s of continuous fall per drop; longer improves terminal-velocity fit |
| Motion | Drop moves monotonically downward; full frame travel is fine |
| Frame count | ≥ 60 frames per drop (rule of thumb: 3 s × FPS) |

Terminal velocity is estimated from a linear fit of vertical position versus time, so any non-gravitational influence (field on, bounce off electrode) invalidates the frames.

### Video B — balanced Brownian motion (field on)

| Property | Requirement |
|---|---|
| Content | Drop(s) levitated or near-stationary; only Brownian fluctuation visible |
| Duration | ≥ 10 × `--max-lag-s` seconds; default max lag is 5 s, so ≥ 50 s recommended — several minutes is better |
| Frame count | ≥ 300 frames at 30 fps for the default 5 s max lag; 3 000 – 6 000 frames gives reliable statistics |
| Graticule | At least one frame must show the reticle grid for scale calibration, or pass `--scale-um-per-px` to skip it |

Only horizontal (`x`) displacement is used for MSD. Any slow residual drift is removed by a per-drop linear fit; large or time-varying drift (airflow, electric-field instability) will still degrade the result.

### Image and optics

| Property | Guideline |
|---|---|
| Droplet size | ≥ 5–8 px diameter in the image; the default trackpy diameter is 11 px and ROI is 60 px — adjust with `--trackpy-diameter-px` and `--roi-size-px` |
| Contrast | Drops must be distinguishable from background; dark drops on bright background (default) or bright on dark (`--particle-polarity bright`) |
| Illumination | Constant; avoid flickering or time-varying background |
| Camera | Fixed position and zoom; no focus drift between frames |
| Scale | Both videos must share the same image scale |

### FPS

Higher FPS gives more displacement samples and smaller statistical error per lag bin. Typical Millikan cameras run at 25–30 fps. The default lag step is 0.5 s (15 frames at 30 fps); very low FPS reduces the number of usable lag points.

## Install

```bash
python -m pip install opencv-python numpy pandas scipy matplotlib tabulate
python -m pip install trackpy
```

`trackpy` is optional. If it is not installed, the script falls back to OpenCV trackers and sub-pixel centroid refinement.

## Quick self-test

Run this before using real videos:

```bash
python millikan_brownian_kb.py --self-test
```

It generates synthetic Brownian/free-fall data, checks the Cunningham radius path, de-drifts horizontal motion, fits MSD, estimates `k_B`, and writes results under `results/self_test/`.

## Calibration for the included videos

Both videos (`particle_1_brown.mov`, `perticle_1_drop.mov`) were recorded with the 世纪中科 Millikan apparatus at the same magnification.  The on-screen reticle has been measured from the first frame:

| Parameter | Value | Source |
|---|---|---|
| Grid cell width/height | 0.2 mm (200 µm) | apparatus spec |
| Grid line spacing | 84 px | measured from frame (peaks at y = 54, 138 … 893) |
| **Image scale** | **200 / 84 ≈ 2.381 µm/px** | derived |
| Frame rate | 10.00 fps | video header |
| Resolution | 1280 × 960 | video header |
| Brownian duration | 479.5 s (~8 min) | video header |
| Free-fall duration | 68.8 s | video header |

The "0" and "1.6" tick marks visible in the right panel span exactly 8 grid rows = 1.6 mm, confirming the 84 px / 0.2 mm calibration.

## Interactive run

### With the included videos

```bash
python millikan_brownian_kb.py \
  --freefall-video perticle_1_drop.mov \
  --brownian-video particle_1_brown.mov \
  --scale-um-per-px 2.381 \
  --fps 10 \
  --temperature-k 293.15 \
  --pressure-pa 101325 \
  --oil-density-kgm3 886 \
  --eta0-pa-s 1.81e-5 \
  --output-dir results/run_001
```

The script will ask you to click one or more Brownian droplets in Video B, then the corresponding free-fall droplet(s) in Video A.  Because `--scale-um-per-px` is provided the calibration step is skipped.

Mouse controls:

- Left click adds a point.
- Right click or `u` removes the last point.
- Enter finishes selection.
- Esc cancels.

### With other videos (scale from graticule)

```bash
python millikan_brownian_kb.py \
  --freefall-video path/to/video_A_freefall.mp4 \
  --brownian-video path/to/video_B_brownian.mp4 \
  --grid-spacing-um 200.0 \
  --grid-intervals 1 \
  --temperature-k 293.15 \
  --pressure-pa 101325 \
  --oil-density-kgm3 886 \
  --eta0-pa-s 1.81e-5 \
  --output-dir results/run_001
```

## Batch/no-GUI run

Use this form when you already know the image scale and droplet positions:

```bash
python millikan_brownian_kb.py \
  --freefall-video perticle_1_drop.mov \
  --brownian-video particle_1_brown.mov \
  --scale-um-per-px 2.381 \
  --fps 10 \
  --brownian-points "315,240" \
  --freefall-points "300,120" \
  --drop-ids "drop1" \
  --temperature-k 293.15 \
  --pressure-pa 101325 \
  --oil-density-kgm3 886 \
  --eta0-pa-s 1.81e-5 \
  --no-gui \
  --output-dir results/run_001
```

Point coordinates are pixel coordinates in the selection frame. Use `--brownian-selection-frame` and `--freefall-selection-frame` if the first frame is not suitable.

## Outputs

The output directory contains:

- `freefall_radius_table.md/csv`: fall time, terminal velocity, Cunningham-corrected radius, slip factor, and viscosity.
- `basic_parameter_table.md/csv`: radius, diffusion coefficient, calculated `k_B`, drift velocity, and fit quality.
- `statistics_table.md/csv`: lag time, sample count, `<dx>`, and `<dx^2>`.
- `msd_fit_table.md/csv`: slopes, intercepts, diffusion constants, and `R^2` values.
- `brownian_trajectory_raw.csv` and `brownian_trajectory_dedrifted.csv`.
- `trajectory_2d.png`, `displacement_histogram.png`, and `msd_fit.png`.
- `run_metadata.json` with parameters, selected points, video metadata, and tracking settings.

## Notes

- Brownian statistics use horizontal `x` displacement only, as requested, to reduce sensitivity to vertical electric-field instability.
- The default viscosity model applies Sutherland's temperature correction to `eta0`; use `--viscosity-model constant` if your `eta0` is already the measured viscosity at experiment temperature.
- If Video A has only one usable free-fall droplet and Video B has multiple Brownian droplets, the script can reuse that single radius for all `k_B` calculations, but matching the same droplet is preferable.
- If the droplet appears bright rather than dark against the background, pass `--particle-polarity bright`.