#!/usr/bin/env python3
"""
Free-fall terminal velocity analysis – Millikan oil drop experiment
Data:  Macintosh HD.txt
Scale: 83.1 units = 0.2 mm
Valid: 183.7 ≤ y ≤ 800  (units)
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt

# ── Physical parameters ───────────────────────────────────────────────────────
ETA0      = 1.81e-5    # Pa·s   dynamic viscosity of air
RHO_OIL   = 886.0     # kg/m³  oil density
RHO_AIR   = 1.204     # kg/m³  air density (20 °C, 1 atm)
G         = 9.80665   # m/s²
PRESSURE  = 101325.0  # Pa
B_CUNN    = 8.2e-3    # Pa·m   Cunningham constant

# ── Scale ─────────────────────────────────────────────────────────────────────
UNITS_REF  = 83.1     # units
DIST_REF   = 0.2e-3   # m  (0.2 mm)
M_PER_UNIT = DIST_REF / UNITS_REF          # m per data-unit
UM_PER_UNIT = M_PER_UNIT * 1e6             # µm per data-unit

Y_MAX = 800.0    # valid upper bound (units)
Y_MIN = 183.7    # valid lower bound (units)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(
    "Macintosh HD.txt",
    sep='\t',
    skiprows=2,          # skip "mass A" header row and column-name row
    names=['t', 'y'],
    na_values=['']
)
df = df.dropna(subset=['y']).reset_index(drop=True)
df['t'] = df['t'].astype(float)
df['y'] = df['y'].astype(float)

# ── Filter to valid range ─────────────────────────────────────────────────────
mask = (df['y'] >= Y_MIN) & (df['y'] <= Y_MAX)
dv   = df[mask].copy().reset_index(drop=True)

print("=" * 60)
print("  FREE-FALL ANALYSIS  (Millikan oil drop)")
print("=" * 60)
print(f"\nTotal data points : {len(df)}")
print(f"Valid data points : {len(dv)}  (183.7 ≤ y ≤ 800 units)")
print(f"Time range        : {dv['t'].min():.2f} – {dv['t'].max():.2f} s")
print(f"y range           : {dv['y'].min():.1f} – {dv['y'].max():.1f} units")
print(f"                  = {dv['y'].min()*UM_PER_UNIT*1e-3:.4f} – "
      f"{dv['y'].max()*UM_PER_UNIT*1e-3:.4f} mm")

# Convert to metres
dv['y_m'] = dv['y'] * M_PER_UNIT

# ── Linear regression (y vs t) ────────────────────────────────────────────────
slope, intercept, r_val, p_val, se = linregress(dv['t'], dv['y_m'])
# slope < 0  (y decreases as particle falls)
v_g  = -slope          # terminal velocity > 0 (downward)
R2   = r_val**2

print(f"\n── Linear Regression ──────────────────────────────────────")
print(f"  Slope     dy/dt = {slope*1e6:.4f} µm/s")
print(f"  Terminal velocity  v_g = {v_g*1e6:.4f} µm/s  "
      f"= {v_g*1e3:.6f} mm/s")
print(f"  R²  = {R2:.6f}    se = {se*1e6:.4f} µm/s")

# ── Stokes law – uncorrected ──────────────────────────────────────────────────
delta_rho = RHO_OIL - RHO_AIR
r0 = np.sqrt(9.0 * ETA0 * v_g / (2.0 * delta_rho * G))
d0 = 2.0 * r0

print(f"\n── Stokes law (no correction) ─────────────────────────────")
print(f"  r₀  = {r0*1e6:.4f} µm")
print(f"  d₀  = {d0*1e6:.4f} µm")

# ── Cunningham slip correction ────────────────────────────────────────────────
# Terminal velocity with Cunningham factor:
#   v_g = [2r²(Δρ)g / (9η)] · (1 + b/(p·r))
# → r² + (b/p)·r − r₀²  =  0
# Positive root:
A_c = 1.0
B_c = B_CUNN / PRESSURE       # b/p  [m]
C_c = -(r0**2)
r_eff = (-B_c + np.sqrt(B_c**2 - 4*A_c*C_c)) / (2*A_c)
d_eff = 2.0 * r_eff

correction_factor = 1.0 + B_CUNN / (PRESSURE * r_eff)

print(f"\n── Cunningham-corrected ───────────────────────────────────")
print(f"  Cunningham factor  (1 + b/pr) = {correction_factor:.4f}")
print(f"  r_eff = {r_eff*1e6:.4f} µm")
print(f"  d_eff = {d_eff*1e6:.4f} µm")

# ── Derived: free-fall distance and time within valid band ───────────────────
dy_m  = (Y_MAX - Y_MIN) * M_PER_UNIT
dt_s  = dy_m / v_g
print(f"\n── Travel across valid band ───────────────────────────────")
print(f"  Δy = {dy_m*1e3:.4f} mm  over  Δt_measured = "
      f"{dv['t'].max()-dv['t'].min():.2f} s")
print(f"  Δt predicted from v_g = {dt_s:.2f} s")

print("\n" + "=" * 60)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Free-fall analysis  –  Millikan oil drop", fontsize=13)

# --- Left: full trajectory ---
ax = axes[0]
ax.plot(df['t'], df['y'] * UM_PER_UNIT * 1e-3,
        'o', color='#cccccc', markersize=3, label='All data')
ax.plot(dv['t'], dv['y'] * UM_PER_UNIT * 1e-3,
        'o', color='steelblue', markersize=4, label='Valid (183.7–800 units)')
t_fit = np.array([dv['t'].min(), dv['t'].max()])
y_fit_mm = (slope * t_fit + intercept) * 1e3
ax.plot(t_fit, y_fit_mm, 'r-', linewidth=2,
        label=f'Fit  $v_g$ = {v_g*1e6:.2f} µm/s\n$R^2$ = {R2:.4f}')
ax.axhline(Y_MAX * UM_PER_UNIT * 1e-3, color='green',  linestyle='--',
           linewidth=1, label='y = 800 units')
ax.axhline(Y_MIN * UM_PER_UNIT * 1e-3, color='orange', linestyle='--',
           linewidth=1, label='y = 183.7 units')
ax.set_xlabel('Time (s)')
ax.set_ylabel('y (mm)')
ax.set_title('Full trajectory')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Right: residuals of valid data ---
ax2 = axes[1]
y_pred_m = slope * dv['t'] + intercept
residuals_um = (dv['y_m'] - y_pred_m) * 1e6
ax2.axhline(0, color='red', linewidth=1)
ax2.plot(dv['t'], residuals_um, 'o', color='steelblue',
         markersize=4, label='Residuals')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Residual (µm)')
ax2.set_title(f'Fit residuals  (σ = {residuals_um.std():.2f} µm)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('freefall_result.png', dpi=150)
plt.show()
print("Plot saved → freefall_result.png")
