"""Chart of the October 2026 live run: prediction minus actual for every talk, frozen model
and live adjustment, from outputs/predictions_log.csv. Writes outputs/oct_2026_errors.png.

Usage: uv run python scripts/plot_oct2026_errors.py
"""
from general_conference_runtime_predictor.paths import OUTPUTS

import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

d = pd.read_csv(OUTPUTS / "predictions_log.csv", encoding="utf-8")
d = d[d.actual_sec.notna() & d.model.isin(["catboost", "catboost_adj"])].copy()
order = ["saturday-morning", "saturday-afternoon", "sunday-morning", "sunday-afternoon"]
d["s"] = d.session.map({s: i for i, s in enumerate(order)})
d = d.sort_values(["s", "speaker_order"])
talks = d[d.model == "catboost"][["session", "speaker_order", "speaker", "s"]].reset_index(drop=True)
talks["x"] = range(1, len(talks) + 1)
key = ["session", "speaker_order"]
d = d.merge(talks[key + ["x"]], on=key)
d["err"] = (d.pred_sec - d.actual_sec) / 60

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)

# session bands + labels
labels = ["Saturday morning", "Saturday afternoon", "Sunday morning", "Sunday afternoon"]
for i, lab in enumerate(labels):
    xs = talks[talks.s == i].x
    if i % 2 == 1:
        ax.axvspan(xs.min() - 0.5, xs.max() + 0.5, color="#f1f0ec", lw=0, zorder=0)
    ax.text((xs.min() + xs.max()) / 2, 6.35, lab, ha="center", va="top", color=INK2, fontsize=10)

ax.axhline(0, color=INK2, lw=1.2, zorder=1)
for m, c, z in [("catboost", BLUE, 3), ("catboost_adj", ORANGE, 4)]:
    g = d[d.model == m]
    ax.scatter(g.x, g.err, s=70, color=c, edgecolor=SURF, linewidth=1.5, zorder=z)

# connect frozen -> adjusted for same talk
adj = d[d.model == "catboost_adj"].set_index("x").err
fro = d[d.model == "catboost"].set_index("x").err
for x in adj.index:
    ax.plot([x, x], [fro[x], adj[x]], color="#c9c8c2", lw=1.2, zorder=2)

# annotations
oaks = d[(d.model == "catboost") & (d.speaker == "Dallin H. Oaks")]
last = oaks[oaks.err > 4].iloc[0]
ax.annotate("President Oaks' closing remarks\n(3:18, predicted 8:52)", (last.x, last.err),
            xytext=(last.x - 13.5, last.err - 0.75), color=INK2, fontsize=9.5,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
first = oaks[oaks.err < 0].iloc[0]
ax.annotate("Oaks' Sunday morning talk,\nthe only one that ran longer than predicted", (first.x, first.err),
            xytext=(first.x - 17.5, first.err - 1.9), color=INK2, fontsize=9.5,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))

ax.set_xlim(0.3, len(talks) + 0.7); ax.set_ylim(-4.6, 6.5)
ax.set_xticks([]); ax.set_xlabel("37 talks, in the order they were given", color=INK2, labelpad=8)
ax.set_ylabel("Prediction minus actual (minutes)", color=INK2)
ax.set_yticks(range(-4, 7)); ax.tick_params(colors=INK2, length=0)
ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
for sp in ax.spines.values(): sp.set_visible(False)

fig.text(0.07, 0.955, "The frozen model ran long on 36 of 37 talks. Adjusting from the day's finished talks fixed most of it.",
         fontsize=13.5, color=INK, weight="bold", ha="left")
fig.text(0.07, 0.915, "General Conference, October 2026. Predictions were logged before each talk started. Mean error: frozen model 1.94 min, adjusted 0.78 min.",
         fontsize=10.5, color=INK2, ha="left")
handles = [Line2D([], [], marker="o", ls="", color=BLUE, ms=9, label="Frozen model (trained on 2010 to April 2026)"),
           Line2D([], [], marker="o", ls="", color=ORANGE, ms=9, label="Adjusted live, using the talks already finished before the current one")]
ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=10, labelcolor=INK2, bbox_to_anchor=(0.0, 0.9))
fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.1)
fig.savefig(OUTPUTS / "oct_2026_errors.png", facecolor=SURF)
print(f"wrote {OUTPUTS / 'oct_2026_errors.png'}")
