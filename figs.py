import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Non-blue categorical palette, validated (lightness, chroma, CVD, contrast all pass)
TERRA, PLUM, OCHRE, GREEN = "#C8551E", "#7A4FA3", "#B58A1A", "#2A8050"
INK, MUTED, GRID = "#222222", "#6B6B6B", "#E3E1DC"

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})
R = json.load(open("results.json"))

# ---------------------------------------------------------------- Fig 1 architecture
fig, ax = plt.subplots(figsize=(7.2, 3.3)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 50)
def box(x, y, w, h, title, lines, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                fc="#FAF8F4", ec=edge, lw=1.3))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=7.6, weight="bold", color=INK)
    for i, l in enumerate(lines):
        ax.text(x + w / 2, y + h - 6.6 - i * 3.5, l, ha="center", va="top", fontsize=6.5, color=INK)
def arrow(x0, y0, x1, y1, label=None, dy=1.4):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, ha="center", fontsize=6.3, color=MUTED)
box(0.5, 27, 19, 21, "Machine M0", ["Primary MPU-6050", "Split-core CT", "DS18B20 on frame"], TERRA)
box(0.5, 1, 19, 21, "Floor near base", ["Reference MPU-6050", "0.3 to 0.6 m from", "machine footing"], OCHRE)
box(28, 1, 30, 47, "Sensor node (edge)", ["ESP32-C3", "1 kHz dual-channel acquisition", "common 32.768 kHz CLKIN", "LMS canceller, N = 32", "1 s features: RMS, kurtosis,", "crest factor, spectral centroid", "context lookup and robust z", "5-window persistence alert", "32-byte summary every 10 s", "immediate packet on alert"], PLUM)
box(66, 1, 18, 47, "Gateway", ["Raspberry Pi 4", "SX1276 radio", "865 to 867 MHz", "commissioning:", "baseline tables", "InfluxDB, MQTT", "OEE (availability)", "runs on factory LAN"], GREEN)
box(89, 1, 10.5, 47, "Users", ["Grafana", "dashboard", "", "Mobile", "alerts", "", "Operator", "part counts"], MUTED)
arrow(19.8, 37, 27.6, 33, "x_p")
arrow(19.8, 11, 27.6, 15, "x_r")
arrow(58.4, 30, 65.6, 30, "LoRa")
arrow(65.6, 18, 58.4, 18, "tables", dy=-4.0)
arrow(84.4, 24.5, 88.6, 24.5)
fig.savefig("fig1_architecture.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 2 residual RMS over time
C = np.load("conv.npz")
FS = 1000
def rms1(x):
    x = x[: x.size // FS * FS].reshape(-1, FS); return x.std(axis=1) * 1000
t0, t1 = 0, 1800
tm = np.arange(t1 - t0) / 60
fig, ax = plt.subplots(figsize=(6.8, 2.8))
ax.plot(tm, rms1(C["xp"] - C["s"])[t0:t1], color=MUTED, lw=1.0, label="Raw primary (no cancellation)")
ax.plot(tm, rms1(C["e_nlms"] - C["s"])[t0:t1], color=TERRA, lw=1.0, label="NLMS without gating")
ax.plot(tm, rms1(C["e_lms"] - C["s"])[t0:t1], color=PLUM, lw=1.2, label="LMS (proposed setting)")
ax.set_xlabel("Time (min)"); ax.set_ylabel("Residual interference,\n1 s RMS (mg)")
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
ax.set_ylim(0, None)
fig.savefig("fig2_trace.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 3 rejection vs leakage
rv = R["rejection_vs_leak"]
leaks = sorted(rv, key=float); x = np.array([float(l) for l in leaks])
fig, ax = plt.subplots(figsize=(5.2, 3.0))
for key, lab, col, mk in [("wiener", "Least-squares bound (32 taps)", MUTED, "s"),
                          ("lms", "LMS, inputs in g, mu = 0.005", PLUM, "o"),
                          ("gnlms", "Gated NLMS", GREEN, "^"),
                          ("nlms", "NLMS without gating", TERRA, "v")]:
    m = np.array([rv[l][key][0] for l in leaks]); s = np.array([rv[l][key][1] for l in leaks])
    ax.errorbar(x, m, yerr=s, color=col, marker=mk, ms=5, lw=1.6, capsize=2, label=lab,
                ls="--" if key == "wiener" else "-")
ax.axhline(0, color=INK, lw=0.6)
ax.set_xlabel("Leakage gain of machine signal into reference")
ax.set_ylabel("Interference rejection (dB)")
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
fig.savefig("fig3_rejection.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 4 detection ablation
reps = R["detection_reps"]
cfgs = [("raw|static", "Raw, static"), ("raw|load_temp", "Raw, context"),
        ("lms|static", "LMS, static"), ("lms|load_temp", "LMS, context")]
fault_keys = [k for k in reps[0]["lms|load_temp"] if "@" in k]
def stat(cfg, key):
    v = [r[cfg][key] for r in reps]; return np.mean(v) * 100, np.std(v) * 100
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.9), gridspec_kw={"width_ratios": [1, 1]})
cols = [MUTED, OCHRE, PLUM, GREEN]
xs = np.arange(len(cfgs))
fp = [stat(c, "fpr") for c, _ in cfgs]
dt = [(np.mean([np.mean([r[c][k] for k in fault_keys]) for r in reps]) * 100,
       np.std([np.mean([r[c][k] for k in fault_keys]) for r in reps]) * 100) for c, _ in cfgs]
for a, vals, title in [(axs[0], fp, "(a) False-positive rate, healthy windows"),
                       (axs[1], dt, "(b) Mean detection rate, faulty windows")]:
    a.bar(xs, [v[0] for v in vals], yerr=[v[1] for v in vals], color=cols, width=0.62,
          capsize=2, edgecolor="white", linewidth=1.5)
    a.set_xticks(xs); a.set_xticklabels([l for _, l in cfgs], rotation=20, ha="right", fontsize=7.5)
    a.set_title(title, fontsize=8.5, loc="left"); a.set_ylabel("%"); a.grid(axis="x", visible=False)
    for xi, v in zip(xs, vals):
        a.text(xi, v[0] + v[1] + 1.0, f"{v[0]:.1f}", ha="center", fontsize=7, color=INK)
fig.tight_layout(); fig.savefig("fig4_detection.png"); plt.close(fig)
print("figs done")
