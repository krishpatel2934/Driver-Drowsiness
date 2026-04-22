# ================================================================
# SCRIPT 03 — Exploratory Data Analysis & Visualisation
# Run on Google Colab AFTER script 02
# Explores the extracted features, saves plots to Drive
# ================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ── Config ───────────────────────────────────────────────────
DRIVE_PROCESSED = '/content/drive/MyDrive/DL_project/processed'
PLOTS_OUT       = os.path.join(DRIVE_PROCESSED, 'plots')
os.makedirs(PLOTS_OUT, exist_ok=True)

LABEL_NAME   = {0: 'Alert', 1: 'Low-vigilant', 2: 'Drowsy'}
LABEL_COLORS = {0: '#3B8BD4', 1: '#EF9F27', 2: '#E24B4A'}

# ── Load master CSV ───────────────────────────────────────────
csv_path = os.path.join(DRIVE_PROCESSED, 'features_all.csv')
print(f"Loading {csv_path}...")
df = pd.read_csv(csv_path)
print(f"Loaded {len(df):,} rows, {df['subject_id'].nunique()} subjects\n")

# ── Basic stats ───────────────────────────────────────────────
print("=" * 50)
print("DATASET SUMMARY")
print("=" * 50)
print(f"Total frames    : {len(df):,}")
print(f"Subjects        : {df['subject_id'].nunique()}")
print(f"Folds           : {df['fold'].unique()}")
print(f"Face detection  : {df['face_detected'].mean()*100:.1f}% of frames")
print(f"\nClass distribution:")
for lbl, cnt in df.groupby('label').size().items():
    pct = cnt / len(df) * 100
    print(f"  {LABEL_NAME[lbl]:15s}: {cnt:>8,} frames ({pct:.1f}%)")

print(f"\nFeature statistics:")
print(df[['ear', 'mar', 'head_tilt', 'perclos']].describe().round(4))

# ── Plot 1: Feature distributions per class ───────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
features  = ['ear', 'mar', 'head_tilt', 'perclos']
titles    = [
    'Eye Aspect Ratio (EAR)',
    'Mouth Aspect Ratio (MAR)',
    'Head Tilt Ratio',
    'PERCLOS (60-sec window)'
]

for ax, feat, title in zip(axes.flat, features, titles):
    for lbl in [0, 1, 2]:
        data = df[df['label'] == lbl][feat].dropna()
        data = data[data > 0]  # exclude no-face frames
        ax.hist(data, bins=80, alpha=0.55,
                color=LABEL_COLORS[lbl],
                label=LABEL_NAME[lbl],
                density=True)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(feat)
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Feature Distributions by Drowsiness Level',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
out = os.path.join(PLOTS_OUT, 'feature_distributions.png')
plt.savefig(out, dpi=130, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out}")

# ── Plot 2: Time-series for one subject ───────────────────────
subject = df['subject_id'].iloc[0]
fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=False)

for lbl in [0, 1, 2]:
    sub_df = df[(df['subject_id'] == subject) & (df['label'] == lbl)]
    t      = sub_df['timestamp_sec'].values
    for ax, feat in zip(axes, ['ear', 'mar', 'head_tilt', 'perclos']):
        ax.plot(t, sub_df[feat].values,
                color=LABEL_COLORS[lbl],
                linewidth=0.6,
                alpha=0.85,
                label=LABEL_NAME[lbl])

feat_labels = [
    'EAR  (↓ = closing)',
    'MAR  (↑ = yawning)',
    'Head tilt (↑ = nodding)',
    'PERCLOS  (↑ = drowsy)'
]
for ax, lbl in zip(axes, feat_labels):
    ax.set_ylabel(lbl, fontsize=10)
    ax.grid(True, alpha=0.25, linewidth=0.5)

axes[0].legend(loc='upper right', fontsize=9)
axes[0].set_title(f'Feature signals over time — Subject {subject}',
                   fontsize=13, fontweight='bold')
axes[-1].set_xlabel('Time (seconds)')

plt.tight_layout()
out = os.path.join(PLOTS_OUT, 'timeseries_subject01.png')
plt.savefig(out, dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

# ── Plot 3: Box plots comparing classes ───────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 5))

for ax, feat, title in zip(axes, features, titles):
    data_by_class = [
        df[(df['label'] == lbl) & (df['face_detected'] == 1)][feat].dropna().values
        for lbl in [0, 1, 2]
    ]
    bp = ax.boxplot(data_by_class,
                    patch_artist=True,
                    notch=True,
                    showfliers=False)
    for patch, lbl in zip(bp['boxes'], [0, 1, 2]):
        patch.set_facecolor(LABEL_COLORS[lbl])
        patch.set_alpha(0.7)
    ax.set_xticklabels([LABEL_NAME[i] for i in [0, 1, 2]],
                        fontsize=9, rotation=15)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Feature Comparison Across Drowsiness Levels',
             fontsize=13, fontweight='bold')
plt.tight_layout()
out = os.path.join(PLOTS_OUT, 'feature_boxplots.png')
plt.savefig(out, dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

# ── Plot 4: Per-subject face detection rate ───────────────────
face_rates = df.groupby('subject_id')['face_detected'].mean() * 100
fig, ax    = plt.subplots(figsize=(12, 4))
colors     = ['#E24B4A' if r < 80 else '#3B8BD4' for r in face_rates.values]
ax.bar(face_rates.index, face_rates.values, color=colors, alpha=0.8)
ax.axhline(80, color='#E24B4A', linewidth=1.5,
           linestyle='--', label='80% threshold')
ax.set_xlabel('Subject ID')
ax.set_ylabel('Face detection rate (%)')
ax.set_title('Face Detection Rate per Subject', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
out = os.path.join(PLOTS_OUT, 'face_detection_rates.png')
plt.savefig(out, dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

print(f"\nAll plots saved to: {PLOTS_OUT}")
print("EDA complete! Run script 04 next.")
