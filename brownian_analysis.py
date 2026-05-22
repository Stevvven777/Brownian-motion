#!/usr/bin/env python3
"""
Brownian motion analysis – Millikan oil drop experiment
Data  : P1_brown.txt   (x position vs time)
Scale : 83.1 units = 0.2 mm  (same as freefall)
Output: brownian_report.pdf
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  USER PARAMETERS  (edit here if experimental conditions differ)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ETA0        = 1.81e-5     # Pa·s   dynamic viscosity of air
TEMPERATURE = 293.15      # K      room temperature
PRESSURE    = 101325.0    # Pa
B_CUNN      = 8.2e-3      # Pa·m   Cunningham constant

# Freefall results (imported from freefall_analysis.py output)
V_G_UM_S    = 48.4542     # µm/s   terminal velocity
R0_UM       = 0.6744      # µm     uncorrected Stokes radius
R_EFF_UM    = 0.6352      # µm     Cunningham-corrected radius

# Scale  (x: 1 unit = 1 mm;  note: freefall y used 83.1 units = 0.2 mm)
M_PER_UNIT  = 1e-3        # m per data-unit
UM_PER_UNIT = M_PER_UNIT * 1e6

# MSD fit range  — use short-lag free-diffusion regime (τ ≤ 20 s)
MAX_LAG_S   = 20.0        # s  (sub-diffusion / confinement sets in beyond ~25 s)
DT          = 0.1         # s  nominal time step

# Reference
K_B_TRUE    = 1.380649e-23  # J/K  (NIST 2018 exact definition)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LOAD & PREPROCESS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
df = pd.read_csv(
    "P1_brown.txt", sep='\t', skiprows=2,
    names=['t', 'x'], na_values=['']
)
df['t'] = df['t'].astype(float)
df['x'] = df['x'].astype(float)

n_nan = df['x'].isna().sum()
# Linear interpolation for isolated NaN values (< 0.5 % of data)
df['x'] = df['x'].interpolate(method='linear')
df = df.dropna().reset_index(drop=True)

# Convert to metres
df['x_m'] = df['x'] * M_PER_UNIT

# ── De-drift (subtract best-fit line) ──────────────────────────────────────
slope_d, intercept_d, r_d, _, _ = linregress(df['t'], df['x_m'])
drift_rate_um_s = slope_d * 1e6
df['x_dd'] = df['x_m'] - (slope_d * df['t'] + intercept_d)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MSD CALCULATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
x = df['x_dd'].values
N = len(x)
max_n = int(MAX_LAG_S / DT)

lags_s = np.arange(1, max_n + 1) * DT
msd_m2 = np.array([
    np.mean((x[n:] - x[:-n])**2)
    for n in range(1, max_n + 1)
])
msd_um2 = msd_m2 * 1e12   # µm²

# ── Fit 1: forced through origin  MSD = 2D·τ ──────────────────────────────
#   argmin Σ(MSD_i - 2D·τ_i)²  →  2D = Σ(τ_i · MSD_i) / Σ(τ_i²)
two_D_fit0 = np.sum(lags_s * msd_m2) / np.sum(lags_s**2)
D_fit0     = two_D_fit0 / 2.0   # m²/s

# ── Fit 2: with intercept (captures noise floor) ──────────────────────────
sl_msd, ic_msd, r_msd, _, se_msd = linregress(lags_s, msd_m2)
D_lr = sl_msd / 2.0             # m²/s

# Primary D: linear-regression with intercept (more robust for real data)
D_primary = D_lr

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  k_B  FROM STOKES–EINSTEIN  D = k_B T / (6π η r)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
r_m = R_EFF_UM * 1e-6
k_B_calc  = 6 * np.pi * ETA0 * r_m * D_primary / TEMPERATURE
k_B_fit0  = 6 * np.pi * ETA0 * r_m * D_fit0    / TEMPERATURE
err_pct   = (k_B_calc - K_B_TRUE) / K_B_TRUE * 100.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONSOLE REPORT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 66)
print("  BROWNIAN MOTION ANALYSIS  –  Millikan oil drop")
print("=" * 66)
print(f"\n  Data points loaded : {len(df)}  (interpolated {n_nan} NaN gaps)")
print(f"  Time span          : 0 – {df['t'].max():.1f} s   Δt = {DT} s\n")

print(f"── De-drift ───────────────────────────────────────────────────────")
print(f"  Drift rate  dx/dt = {drift_rate_um_s:.4f} µm/s")
print(f"  (R² of drift fit  = {r_d**2:.4f})\n")

print(f"── MSD fit (τ = 0.1 – {MAX_LAG_S:.0f} s) ──────────────────────────────────")
print(f"  Fit (origin forced) :  2D = {two_D_fit0*1e12:.4f} µm²/s")
print(f"                          D = {D_fit0*1e12:.4f} µm²/s = {D_fit0:.4e} m²/s")
print(f"  Fit (with intercept):  2D = {sl_msd*1e12:.4f} µm²/s")
print(f"                          D = {D_lr*1e12:.4f} µm²/s = {D_lr:.4e} m²/s")
print(f"  intercept = {ic_msd*1e12:.4f} µm²    R² = {r_msd**2:.5f}\n")

print(f"── Particle (from freefall) ───────────────────────────────────────")
print(f"  v_g               = {V_G_UM_S:.4f} µm/s")
print(f"  r₀  (Stokes)      = {R0_UM:.4f} µm")
print(f"  r_eff (Cunningham)= {R_EFF_UM:.4f} µm   d_eff = {2*R_EFF_UM:.4f} µm\n")

print(f"── Boltzmann constant (Stokes–Einstein) ───────────────────────────")
print(f"  k_B (origin fit)  = {k_B_fit0:.4e} J/K")
print(f"  k_B (linear fit)  = {k_B_calc:.4e} J/K   ← primary result")
print(f"  k_B (NIST exact)  = {K_B_TRUE:.4e} J/K")
print(f"  Relative error    = {err_pct:+.2f} %\n")
print("=" * 66)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FIGURE / REPORT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig = plt.figure(figsize=(14, 11))
fig.patch.set_facecolor('#fafafa')
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

BLUE  = '#2c6fad'
GRAY  = '#aaaaaa'
RED   = '#d62728'
GREEN = '#2ca02c'

# ── Panel A: raw x trajectory ──────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, :])   # full-width top row
ax0.plot(df['t'], df['x_m'] * 1e6, lw=0.6, color=BLUE, alpha=0.8, label='Raw x')
t_line = np.array([df['t'].min(), df['t'].max()])
ax0.plot(t_line,
         (slope_d * t_line + intercept_d) * 1e6,
         color=RED, lw=1.8, ls='--',
         label=f'Drift  ({drift_rate_um_s:.3f} µm/s)')
ax0.set_xlabel('Time  (s)', fontsize=10)
ax0.set_ylabel('x  (µm)', fontsize=10)
ax0.set_title('(A)  Raw Brownian trajectory  +  drift fit', fontsize=11, fontweight='bold')
ax0.legend(fontsize=9)
ax0.grid(True, alpha=0.25)

# ── Panel B: de-drifted trajectory ────────────────────────────────────────
ax1 = fig.add_subplot(gs[1, 0])
ax1.plot(df['t'], df['x_dd'] * 1e6, lw=0.5, color=BLUE, alpha=0.75)
ax1.axhline(0, color='k', lw=0.8, ls=':')
ax1.set_xlabel('Time  (s)', fontsize=10)
ax1.set_ylabel('x  (µm)', fontsize=10)
ax1.set_title('(B)  De-drifted trajectory', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.25)

# ── Panel C: displacement histogram ───────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 1])
dx1 = (df['x_dd'].diff().dropna() * 1e6).values   # 1-step displacements µm
bins = np.linspace(dx1.min(), dx1.max(), 60)
ax2.hist(dx1, bins=bins, color=BLUE, alpha=0.7, edgecolor='white', lw=0.4,
         density=True, label='1-step Δx')
# Gaussian overlay
from scipy.stats import norm
mu_dx, sig_dx = norm.fit(dx1)
xg = np.linspace(dx1.min(), dx1.max(), 300)
ax2.plot(xg, norm.pdf(xg, mu_dx, sig_dx), color=RED, lw=2,
         label=f'Gaussian σ={sig_dx:.3f} µm')
ax2.set_xlabel('Δx per step  (µm)', fontsize=10)
ax2.set_ylabel('Probability density', fontsize=10)
ax2.set_title('(C)  Step-displacement distribution', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.25)

# ── Panel D: MSD vs τ ─────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2, 0])
ax3.scatter(lags_s, msd_um2, s=6, color=BLUE, alpha=0.6, label='MSD data', zorder=3)

# Fit lines
t_f = np.array([0, MAX_LAG_S])
ax3.plot(t_f,  two_D_fit0 * 1e12 * t_f,
         color=GREEN, lw=2, label=f'2D·τ (origin forced)\nD={D_fit0*1e12:.3f} µm²/s')
ax3.plot(t_f,  (sl_msd * 1e12 * t_f + ic_msd * 1e12),
         color=RED, lw=2, ls='--',
         label=f'Linear fit (w/ intercept)\nD={D_lr*1e12:.3f} µm²/s  R²={r_msd**2:.4f}')

ax3.set_xlabel('Lag τ  (s)', fontsize=10)
ax3.set_ylabel('MSD  (µm²)', fontsize=10)
ax3.set_title('(D)  Mean Square Displacement', fontsize=11, fontweight='bold')
ax3.legend(fontsize=7.5)
ax3.grid(True, alpha=0.25)
ax3.set_xlim(0, MAX_LAG_S)
ax3.set_ylim(0)

# ── Panel E: results summary ──────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 1])
ax4.axis('off')

lines = [
    ("FREEFALL",                 None),
    ("Terminal velocity  vg",   f"{V_G_UM_S:.4f}  µm/s"),
    ("Radius r₀ (Stokes)",      f"{R0_UM:.4f}  µm"),
    ("Radius r_eff (Cunningham)",f"{R_EFF_UM:.4f}  µm"),
    ("Diameter d_eff",           f"{2*R_EFF_UM:.4f}  µm"),
    ("",                         None),
    ("BROWNIAN MOTION",          None),
    ("Drift rate  dx/dt",        f"{drift_rate_um_s:.4f}  µm/s"),
    ("D (origin fit)",           f"{D_fit0*1e12:.4f}  µm²/s"),
    ("D (linear fit) ★",        f"{D_lr*1e12:.4f}  µm²/s"),
    ("",                         None),
    ("BOLTZMANN CONSTANT",       None),
    ("T",                        f"{TEMPERATURE:.2f}  K"),
    ("η",                        f"{ETA0:.2e}  Pa·s"),
    ("k_B  (origin fit)",        f"{k_B_fit0:.4e}  J/K"),
    ("k_B  (linear fit) ★",     f"{k_B_calc:.4e}  J/K"),
    ("k_B  (NIST exact)",        f"{K_B_TRUE:.4e}  J/K"),
    ("Relative error",           f"{err_pct:+.2f}  %"),
]

y0 = 0.97
dy = 0.97 / len(lines)
for label, val in lines:
    if val is None:   # section header
        ax4.text(0.02, y0, label, transform=ax4.transAxes,
                 fontsize=9.5, fontweight='bold', color='#333333',
                 va='top')
    else:
        ax4.text(0.02, y0, label + ":", transform=ax4.transAxes,
                 fontsize=8.5, color='#555555', va='top')
        ax4.text(0.98, y0, val, transform=ax4.transAxes,
                 fontsize=8.5, color='#111111', va='top', ha='right',
                 fontfamily='monospace')
    y0 -= dy

ax4.set_title('(E)  Summary of results', fontsize=11, fontweight='bold')

# ── Title ─────────────────────────────────────────────────────────────────
fig.suptitle(
    "Millikan oil drop experiment  —  Brownian motion & free-fall analysis",
    fontsize=13, fontweight='bold', y=0.995
)

plt.savefig('brownian_report.pdf', dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.savefig('brownian_report.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("Report saved → brownian_report.pdf  /  brownian_report.png")
