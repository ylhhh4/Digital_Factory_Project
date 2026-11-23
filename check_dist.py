import glob, os
import numpy as np
import matplotlib.pyplot as plt

# Image resolution (1920x1080 based on your previous JSON records; modify if changed)
IMG_W, IMG_H = 1920, 1080
label_files = glob.glob("11106655_pallet_part00_phase20/labels/*.txt")

cxs, cys, areas, short_sides_px = [], [], [], []

for f in label_files:
    with open(f, "r") as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = map(float, parts[:5])
            cxs.append(cx)
            cys.append(cy)
            areas.append(w * h * 100)  # Percentage of image area (%)
            short_sides_px.append(min(w * IMG_W, h * IMG_H))

areas = np.array(areas)
short_sides_px = np.array(short_sides_px)

# 1. Print key statistics and R9 rule check
print(f"Total valid bounding boxes: {len(areas)}")
print(f"Area ratio (%): Min={areas.min():.2f}%, Median={np.median(areas):.2f}%, Max={areas.max():.2f}%")
print(f"Scale distribution: Small (<1%)={(areas < 1).sum()} | Medium (1%~10%)={((areas >= 1) & (areas < 10)).sum()} | Large (>=10%)={(areas >= 10).sum()}")
bad_r9 = (short_sides_px < 16).sum()
print(f"R9 rule check (short side < 16px): {bad_r9} violations (shortest side is {short_sides_px.min():.1f} px)")

# 2. Plot distribution charts
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left plot: Area distribution histogram
axes[0].hist(areas, bins=20, color="#4C72B0", edgecolor="black")
axes[0].set_title("Box Area Distribution (% of Image)")
axes[0].set_xlabel("Area (%)")
axes[0].set_ylabel("Count")

# Right plot: Center point scatter plot (note: image coordinate system has y-axis pointing downward)
axes[1].scatter(cxs, cys, alpha=0.6, color="#DD8452", edgecolors="k")
axes[1].set_xlim(0, 1)
axes[1].set_ylim(1, 0)
axes[1].set_title("Box Center Position (cx, cy)")
axes[1].set_xlabel("cx (0=Left, 1=Right)")
axes[1].set_ylabel("cy (0=Top, 1=Bottom)")
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("distribution_check.png", dpi=150)
print("Distribution plot saved as: distribution_check.png")
