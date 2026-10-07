"""Charts of the October 2026 live run: prediction minus actual for every talk, frozen model
and live adjustment, from outputs/predictions_log.csv.

Writes outputs/oct_2026_errors.png (wide, for the README) and
outputs/oct_2026_errors_square.png (1080x1080 with larger type, for social posts).

Usage: uv run python scripts/plot_oct2026_errors.py
"""
from general_conference_runtime_predictor.paths import OUTPUTS

import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

SURF, INK, INK2, GRID, BAND, STEM = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#f1f0ec", "#c9c8c2"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SESSIONS = ["saturday-morning", "saturday-afternoon", "sunday-morning", "sunday-afternoon"]
LABELS = ["Saturday morning", "Saturday afternoon", "Sunday morning", "Sunday afternoon"]


def load():
    d = pd.read_csv(OUTPUTS / "predictions_log.csv", encoding="utf-8")
    d = d[d.actual_sec.notna() & d.model.isin(["catboost", "catboost_adj"])].copy()
    d["s"] = d.session.map({s: i for i, s in enumerate(SESSIONS)})
    d = d.sort_values(["s", "speaker_order"])
    talks = d[d.model == "catboost"][["session", "speaker_order", "speaker", "s"]].reset_index(drop=True)
    talks["x"] = range(1, len(talks) + 1)
    d = d.merge(talks[["session", "speaker_order", "x"]], on=["session", "speaker_order"])
    d["err"] = (d.pred_sec - d.actual_sec) / 60
    return d, talks


def draw(ax, d, talks, fs, dot, band_label_y, first_band_note=None, two_line_labels=False):
    """Dots, stems, session bands. `fs` is the base font size, `dot` the marker area."""
    for i, lab in enumerate(LABELS):
        xs = talks[talks.s == i].x
        if i % 2 == 1:
            ax.axvspan(xs.min() - 0.5, xs.max() + 0.5, color=BAND, lw=0, zorder=0)
        text = lab.replace(" ", "\n") if two_line_labels else lab
        ax.text((xs.min() + xs.max()) / 2, band_label_y, text, ha="center", va="top", color=INK2, fontsize=fs,
                linespacing=1.15)
    if first_band_note:
        xs = talks[talks.s == 0].x
        ax.text((xs.min() + xs.max()) / 2, -0.35, first_band_note, ha="center", va="top", color=INK2,
                fontsize=fs - 1.5, style="italic", linespacing=1.3)
    ax.axhline(0, color=INK2, lw=1.2, zorder=1)
    adj = d[d.model == "catboost_adj"].set_index("x").err
    fro = d[d.model == "catboost"].set_index("x").err
    for x in adj.index:
        ax.plot([x, x], [fro[x], adj[x]], color=STEM, lw=1.4, zorder=2)
    for m, c, z in [("catboost", BLUE, 3), ("catboost_adj", ORANGE, 4)]:
        g = d[d.model == m]
        ax.scatter(g.x, g.err, s=dot, color=c, edgecolor=SURF, linewidth=1.5, zorder=z)
    ax.set_xlim(0.3, len(talks) + 0.7)
    ax.set_xticks([])
    ax.tick_params(colors=INK2, length=0, labelsize=fs)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return fro, adj


def wide(d, talks):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    fro, adj = draw(ax, d, talks, fs=10, dot=70, band_label_y=6.35)
    oaks = d[(d.model == "catboost") & (d.speaker == "Dallin H. Oaks")]
    last = oaks[oaks.err > 4].iloc[0]
    ax.annotate("President Oaks' closing remarks\n(3:18, predicted 8:52)", (last.x, last.err),
                xytext=(last.x - 13.5, last.err - 0.75), color=INK2, fontsize=9.5,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    first = oaks[oaks.err < 0].iloc[0]
    ax.annotate("Oaks' Sunday morning talk,\nthe only one that ran longer than predicted", (first.x, first.err),
                xytext=(first.x - 17.5, first.err - 1.9), color=INK2, fontsize=9.5,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_ylim(-4.6, 6.5); ax.set_yticks(range(-4, 7))
    ax.set_xlabel("37 talks, in the order they were given", color=INK2, labelpad=8)
    ax.set_ylabel("Prediction minus actual (minutes)", color=INK2)
    fig.text(0.07, 0.955, "The frozen model ran long on 36 of 37 talks. Adjusting from the day's finished talks fixed most of it.",
             fontsize=13.5, color=INK, weight="bold", ha="left")
    fig.text(0.07, 0.915, "General Conference, October 2026. Predictions were logged before each talk started. "
             "Mean error: frozen model 1.94 min, adjusted 0.78 min over the 29 talks it covered.",
             fontsize=10.5, color=INK2, ha="left")
    handles = [Line2D([], [], marker="o", ls="", color=BLUE, ms=9, label="Frozen model (trained on 2010 to April 2026)"),
               Line2D([], [], marker="o", ls="", color=ORANGE, ms=9,
                      label="Adjusted live, using the talks already finished before the current one")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=10, labelcolor=INK2, bbox_to_anchor=(0.0, 0.9))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.1)
    out = OUTPUTS / "oct_2026_errors.png"
    fig.savefig(out, facecolor=SURF)
    print(f"wrote {out}")


def square(d, talks):
    """1080x1080 version for a feed: bigger type, one annotation, a note on the empty first band."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15})
    fig, ax = plt.subplots(figsize=(9, 9), dpi=120)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    fro, adj = draw(ax, d, talks, fs=14, dot=120, band_label_y=6.45, two_line_labels=True)
    oaks = d[(d.model == "catboost") & (d.speaker == "Dallin H. Oaks")]
    last = oaks[oaks.err > 4].iloc[0]
    ax.annotate("President Oaks' 3-minute\nclosing remarks\n(predicted 8:52)", (last.x, last.err),
                xytext=(last.x - 15, 3.85), color=INK2, fontsize=13, va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9, shrinkB=6))
    first = oaks[oaks.err < 0].iloc[0]
    ax.annotate("President Oaks' Sunday morning\nsermon, the only talk that ran\nlonger than predicted", (first.x, first.err),
                xytext=(first.x - 17.5, -3.1), color=INK2, fontsize=13, va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9, shrinkB=6))
    ax.set_ylim(-4.6, 6.5); ax.set_yticks(range(-4, 7, 2))
    ax.set_xlabel("37 talks, in the order they were given", color=INK2, labelpad=10, fontsize=14)
    # Direction cues instead of a single axis label: one per half of the axis.
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f} min" if v else "0"))
    cue = dict(rotation=90, ha="center", va="center", color=INK2, fontsize=12.5, transform=ax.get_yaxis_transform())
    ax.text(-0.14, 3.2, "▲  model predicted LONGER", **cue)
    ax.text(-0.14, -2.3, "▼  model predicted SHORTER", **cue)
    fig.text(0.05, 0.955, "The model ran long on 36 of 37 talks.", fontsize=21, color=INK, weight="bold", ha="left")
    fig.text(0.05, 0.915, "Adjusting from finished talks fixed most of it.", fontsize=21, color=INK, weight="bold", ha="left")
    fig.text(0.05, 0.875, "General Conference, October 2026. Every prediction was logged before the talk started.\n"
             "Mean miss: 1.9 min before the adjustment, 0.8 min after (over the 29 talks it covered).",
             fontsize=13, color=INK2, ha="left", va="top", linespacing=1.4)
    handles = [Line2D([], [], marker="o", ls="", color=BLUE, ms=12, label="Model trained before the weekend"),
               Line2D([], [], marker="o", ls="", color=ORANGE, ms=12, label="Adjusted live from finished talks")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=13.5, labelcolor=INK2,
              bbox_to_anchor=(0.0, 0.905), handletextpad=0.4, borderaxespad=0.2)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.79, bottom=0.08)
    out = OUTPUTS / "oct_2026_errors_square.png"
    fig.savefig(out, facecolor=SURF)
    print(f"wrote {out}")


if __name__ == "__main__":
    data, talk_table = load()
    wide(data, talk_table)
    square(data, talk_table)
