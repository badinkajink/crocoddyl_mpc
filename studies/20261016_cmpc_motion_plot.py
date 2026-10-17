#!/usr/bin/env python3
"""Plot saved startup traces with system Python; no model import or integration."""
import csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'studies/runs/20261016-cmpc-motion-transfer'
MEDIA=ROOT/'docs/lean/media/20261015-contact-audit'
CASES=['baseline','fixed-position','velocity']
arrays={}
for case in CASES:
    stem=OUT/('20261016-'+case)
    with Path(str(stem)+'.csv').open() as f:
        reader=csv.reader(x for x in f if not x.startswith('#'))
        names=next(reader); rows=np.array(list(reader),float)
    arrays[case]=(dict(zip(names,rows.T)),dict(np.load(str(stem)+'-raw.npz')))
# Standalone trace figure; supports separation of command and contact transients.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
for case,color in zip(CASES,['#2266aa','#777777','#d97800']):
    c,a=arrays[case];servo=dict(np.load(OUT/('20261016-'+case+'-servo.npz')));t=a['t'];mask=t<.5
    j=list(servo['joint_names']).index('left_hip_pitch_joint')
    axes[0,0].plot(t[mask],servo['v'][mask,j],color=color,label=case)
    axes[0,0].plot(t[mask],servo['v_des'][mask,j],color=color,ls='--',alpha=.6)
    acc=np.max(abs(np.diff(servo['v'],axis=0)/np.diff(t)[:,None]),axis=1);amask=t[1:]<.5
    axes[0,1].plot((t[1:]+t[:-1])[amask]/2,acc[amask],color=color)
    axes[1,0].plot(t[mask],servo['ff'][mask,j],color=color)
    axes[1,1].plot(t[mask],np.linalg.norm(a['point_v'][mask,0],axis=1),color=color)
    axes[2,0].plot(t[mask],a['sole_w'][mask,:,2].sum(axis=1),color=color)
    axes[2,1].plot(t[mask],np.max(abs(servo['requested_torque'][mask]-servo['actuator_torque'][mask]),axis=1),color=color)
for ax,label in zip(axes.flat,['Left hip speed (rad/s), dashed: command','Maximum joint acceleration (rad/s²)','Left hip feedforward (N m)','Wrist speed (m/s)','Total floor normal force (N)','Torque clipping difference (N m)']):
    ax.set_ylabel(label);ax.grid(alpha=.2);ax.set_xlim(0,.5)
axes[0,0].legend();axes[2,0].set_xlabel('Simulation time (s)');axes[2,1].set_xlabel('Simulation time (s)')
fig.tight_layout();fig.savefig(MEDIA/'20261016-motion-startup-traces.png',dpi=160);plt.close(fig)
