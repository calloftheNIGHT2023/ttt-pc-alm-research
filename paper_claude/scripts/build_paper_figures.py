from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "confirmation-key-contrasts.png"
MECHANISM_FULL = ROOT / "figures" / "mechanism-witness.png"
MECHANISM_ESCAPE = ROOT / "figures" / "mechanism-escape.png"

# Values are copied from the frozen 8,192-task confirmation report.
# Difference is primary minus control; negative favors the primary method.
rows = [
    ("Archive only", -0.002188720661, -0.002518017629, -0.001865434101, -0.001718500368),
    ("Ordinary PC", -0.001484161731, -0.001794663813, -0.001179734456, -0.001042235502),
    ("No dual", -0.001575840633, -0.001858519237, -0.001300355152, -0.001179066439),
    ("Adam 60", -0.000055564991, -0.000350296342, 0.000237477836, 0.000376452193),
    ("Adam 120", -0.000034726878, -0.000326595714, 0.000255885251, 0.000395675547),
    ("Adam 240", -0.000029947868, -0.000322913518, 0.000261232133, 0.000400454486),
    ("Adam 480", -0.000025610286, -0.000318054241, 0.000264401156, 0.000404153882),
    ("Plain ALM 64", -0.002940045732, -0.003357439482, -0.002527844555, -0.002342447370),
    ("Plain ALM 128", -0.001900211106, -0.002294668223, -0.001511927676, -0.001327432189),
]

labels = [row[0] for row in rows][::-1]
mean = np.array([row[1] for row in rows][::-1]) * 1e3
lo = np.array([row[2] for row in rows][::-1]) * 1e3
hi = np.array([row[3] for row in rows][::-1]) * 1e3
upper = np.array([row[4] for row in rows][::-1]) * 1e3
passed = upper < 0
y = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
ax.axvline(0, color="#b94a48", linestyle="--", linewidth=1.2)
for i in range(len(labels)):
    color = "#1b877c" if passed[i] else "#7e848b"
    ax.plot([lo[i], hi[i]], [y[i], y[i]], color=color, linewidth=2.2)
    ax.scatter(mean[i], y[i], s=42, color=color, zorder=3)
    ax.scatter(upper[i], y[i], s=54, marker=">", color=color, zorder=3)

ax.set_yticks(y, labels)
ax.set_xlabel(r"Primary minus control query MSE ($\times 10^{-3}$; negative favors primary)")
ax.set_title("Frozen confirmation: selected preregistered contrasts")
ax.grid(axis="x", alpha=0.2)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

fig.savefig(OUT, dpi=240, facecolor="white")
plt.close(fig)

with Image.open(MECHANISM_FULL) as image:
    image.crop((0, 0, image.width, 760)).save(MECHANISM_ESCAPE)
