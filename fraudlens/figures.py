"""Static figures for the README, the solution document and the pitch deck.

One visual language everywhere: thin marks, a hairline grid, the champion in
blue and comparators in grey, values labelled directly on the marks.
"""

import json

import matplotlib.pyplot as plt
import pandas as pd

from . import config

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GREY, CRITICAL = "#2a78d6", "#eb6834", "#b9b8b1", "#d03b3b"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["Segoe UI", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def _save(fig, name: str) -> None:
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES / name)
    plt.close(fig)


def model_comparison(metrics: dict) -> None:
    """Frauds caught in the day-2 top 200, baselines against models."""
    k = metrics["capacity"]
    rows = [
        ("Review 200 at random", metrics["baselines"]["random_review"]["frauds_in_queue"]),
        ("Review the 200 largest amounts", metrics["baselines"]["largest_amount_first"]["frauds_in_queue"]),
    ]
    rows += [(name, m["day2"]["frauds_in_queue"]) for name, m in metrics["models"].items()]
    rows.sort(key=lambda r: r[1])
    champion = metrics["champion"]["name"]

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    colours = [BLUE if name == champion else GREY for name, _ in rows]
    bars = ax.barh([name for name, _ in rows], [value for _, value in rows], color=colours, height=0.55)
    for bar, (name, value) in zip(bars, rows, strict=True):
        label = f"{value:.0f}" if value >= 1 else f"{value:.1f}"
        ax.text(
            bar.get_width() + 2,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            color=INK,
            fontweight="bold" if name == champion else "normal",
        )
    total = metrics["champion"]["day2"]["frauds_total"]
    ax.axvline(total, color=MUTED, linewidth=1)
    ax.text(total - 2, len(rows) - 0.5, f"all {total} frauds on day 2", ha="right", va="center", color=MUTED, fontsize=9)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.set_xlim(0, total * 1.06)
    ax.set_xlabel(f"Real frauds among the {k} alerts reviewed (day 2, unseen)")
    ax.set_title(f"Frauds caught with {k} reviews a day")
    ax.grid(axis="y", visible=False)
    _save(fig, "model_comparison.png")


def pr_curve(metrics: dict) -> None:
    """Precision-recall curves on day 2, with the 200-alert operating point marked."""
    curves = pd.read_csv(config.CURVES_CSV)
    pr = curves[curves["curve"] == "pr"]
    champion = metrics["champion"]["name"]
    point = metrics["champion"]["day2"]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, colour in (("logistic_balanced", GREY), (champion, BLUE)):
        part = pr[pr["model"] == name].sort_values("x")
        ax.plot(part["x"], part["y"], color=colour, linewidth=2, label=f"{name} (PR-AUC {metrics['models'][name]['day2']['pr_auc']:.2f})")
    ax.scatter([point["recall"]], [point["precision"]], s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.annotate(
        f"200 alerts/day\nprecision {point['precision']:.0%}, recall {point['recall']:.0%}",
        (point["recall"], point["precision"]), xytext=(-14, -44), textcoords="offset points", ha="right", color=INK,
    )  # fmt: skip
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("Recall: share of all frauds caught")
    ax.set_ylabel("Precision: share of alerts that are fraud")
    ax.set_title("Precision-recall on the unseen day")
    ax.legend(loc="lower left")
    _save(fig, "pr_curve.png")


def capacity_curve(metrics: dict) -> None:
    """What each extra review buys: precision and recall against alerts per day."""
    curve = pd.read_csv(config.OUTPUTS / "capacity_curve.csv")
    curve = curve[curve["alerts_per_day"] <= 600]
    k = metrics["capacity"]

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(curve["alerts_per_day"], curve["recall"], color=BLUE, linewidth=2, label="Recall (frauds caught)")
    ax.plot(curve["alerts_per_day"], curve["precision"], color=ORANGE, linewidth=2, label="Precision (alerts that are fraud)")
    ax.axvline(k, color=MUTED, linewidth=1)
    ax.text(k + 8, 0.04, f"today's capacity: {k}", color=MUTED, fontsize=9)
    last = curve.iloc[-1]
    ax.text(last["alerts_per_day"] + 6, last["recall"], f"{last['recall']:.0%}", color=INK, va="center")
    ax.text(last["alerts_per_day"] + 6, last["precision"], f"{last['precision']:.0%}", color=INK, va="center")
    ax.set_xlim(0, 640)
    ax.set_ylim(0, 1.04)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Alerts reviewed per day")
    ax.set_title("Capacity planner: what more reviews buy")
    ax.legend(loc="center right")
    _save(fig, "capacity_curve.png")


def leakage_exhibit(metrics: dict) -> None:
    """The same model scored two ways: SMOTE before the split against after it."""
    exhibit = metrics["leakage_exhibit"]
    rows = [
        ("SMOTE after the split\n(correct)", exhibit["smote_after_split"]["pr_auc"], BLUE),
        ("SMOTE before the split\n(leaks)", exhibit["smote_before_split"]["pr_auc"], CRITICAL),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    bars = ax.barh([r[0] for r in rows], [r[1] for r in rows], color=[r[2] for r in rows], height=0.5)
    for bar, (_, value, _) in zip(bars, rows, strict=True):
        ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", color=INK, fontweight="bold")
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("PR-AUC reported on the test set")
    ax.set_title("Where SMOTE sits changes the score, not the model")
    ax.grid(axis="y", visible=False)
    _save(fig, "leakage_exhibit.png")


def render_all() -> None:
    plt.switch_backend("Agg")  # write files only; never open a window
    style()
    metrics = json.loads(config.METRICS_JSON.read_text())
    for figure in (model_comparison, pr_curve, capacity_curve, leakage_exhibit):
        figure(metrics)
    print(f"[figures] written to {config.FIGURES}")
