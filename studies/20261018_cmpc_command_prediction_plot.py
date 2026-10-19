#!/usr/bin/env python3
"""Plot saved predictor errors with system Python; use resguard."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'studies/runs/20261018-cmpc-model-uncertainty'
MEDIA = ROOT/'docs/lean/media/20261015-contact-audit'
fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
labels = ['Frozen torque, 20 ms', 'Frozen torque, 2 ms', 'Servo law, 2 ms']
variants = ['frozen_20ms', 'frozen_2ms', 'servo_2ms']
for row, case in enumerate(['nominal', 'pulse']):
    a = np.load(OUT/f'20261018-{case}-prediction.npz')
    for variant, label in zip(variants, labels):
        error = np.linalg.norm(a[variant+'_residual'][:, 36:], axis=1)
        axes[row, 0].plot(a['time'], error, label=label, linewidth=1)
    rows = json.loads((OUT/f'20261018-{case}-endpoint-rows.json').read_text())
    excess = {}
    for r in rows:
        excess[r['t']] = max(excess.get(r['t'], 0.), max(r['cone_violation'][:5]))
    axes[row, 1].plot(list(excess), list(excess.values()), color='#943c17')
    for col in range(2):
        axes[row, col].set_xlabel('Simulation time (s)')
        axes[row, col].grid(alpha=.2)
    axes[row, 0].set_title(case.capitalize()+' saved transitions')
    axes[row, 0].set_ylabel('Velocity residual norm (rad/s)')
    axes[row, 1].set_title(case.capitalize()+' previous-command endpoint')
    axes[row, 1].set_ylabel('Maximum force-row excess (N)')
axes[0, 0].legend(fontsize=8)
fig.savefig(MEDIA/'20261018-model-uncertainty-prediction.png', dpi=150)
plt.close(fig)
