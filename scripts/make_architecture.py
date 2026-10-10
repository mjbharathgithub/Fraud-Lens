"""Draw docs/architecture.png: the four stages of FraudLens as a snake-shaped flow.

python scripts/make_architecture.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "architecture.png"

SURFACE, BAND, INK, INK_2, MUTED, BORDER, BLUE = "#fcfcfb", "#f1f0ec", "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#2a78d6"

# Each row: (stage label, direction, [(title, detail), ...]). Rows alternate direction so the flow snakes down the page.
ROWS = [
    (
        "1  DATA",
        +1,
        [
            ("Public dataset", "284,807 card transactions\nover two days"),
            ("Validate", "schema, missing values,\nfile checksum"),
            ("De-duplicate", "1,081 exact copies removed\nbefore any split"),
            ("Split by day", "day 1 = training\nday 2 = unseen test"),
        ],
    ),
    (
        "2  LEARN  (day 1 only)",
        -1,
        [
            ("Feature pipeline", "log amount, hour of day,\nV1-V28 (inside the Pipeline)"),
            ("Candidates + baselines", "logistic, 3 x XGBoost,\n5-fold cross-validation"),
            ("MLflow", "every run tracked,\ncompared and registered"),
            ("Champion bundle", "model + reference stats,\none small file"),
        ],
    ),
    (
        "3  TRIAGE  (every day)",
        +1,
        [
            ("Score and rank", "CSV batch or /score API;\nkeep the top 200 = capacity"),
            ("Reasons", "exact SHAP contributions,\ntop 3 per alert"),
            ("Action policy", "rules by risk band:\nhold, step-up or review"),
            ("Review-note agent", "template, or local LLM with\na check on every number"),
        ],
    ),
    (
        "4  DECIDE  (human in the loop)",
        -1,
        [
            ("Case store", "every draft waits as\nPENDING_APPROVAL"),
            ("Analyst console", "Streamlit: approve,\nreject or escalate"),
            ("Audit log", "who decided what,\nand when"),
            ("Feedback labels", "confirmed outcomes feed\nthe next training run"),
        ],
    ),
]

BOX_W, BOX_H, GAP_X, ROW_H, LEFT = 2.75, 1.05, 0.55, 1.95, 0.35


def main() -> None:
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"]})
    width = LEFT * 2 + 4 * BOX_W + 3 * GAP_X
    height = len(ROWS) * ROW_H + 0.3
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    centres = {}
    for r, (label, direction, boxes) in enumerate(ROWS):
        top = height - 0.15 - r * ROW_H
        ax.add_patch(
            FancyBboxPatch(
                (0.1, top - ROW_H + 0.12),
                width - 0.2,
                ROW_H - 0.24,
                boxstyle="round,pad=0,rounding_size=0.12",
                facecolor=BAND,
                edgecolor="none",
            )
        )
        # The label sits on the side away from the arrow arriving from the row above.
        if direction > 0 and r > 0:
            ax.text(width - LEFT, top - 0.36, label, fontsize=10.5, fontweight="bold", color=INK_2, va="center", ha="right")
        else:
            ax.text(LEFT, top - 0.36, label, fontsize=10.5, fontweight="bold", color=INK_2, va="center")
        for i, (title, detail) in enumerate(boxes):
            column = i if direction > 0 else 3 - i
            x = LEFT + column * (BOX_W + GAP_X)
            y = top - 0.62 - BOX_H
            human_gate = title in ("Case store", "Analyst console")
            ax.add_patch(
                FancyBboxPatch(
                    (x, y), BOX_W, BOX_H, boxstyle="round,pad=0,rounding_size=0.1",
                    facecolor="white", edgecolor=BLUE if human_gate else BORDER, linewidth=1.8 if human_gate else 1,
                )
            )  # fmt: skip
            ax.text(x + BOX_W / 2, y + BOX_H - 0.27, title, fontsize=11, fontweight="bold", color=INK, ha="center", va="center")
            ax.text(x + BOX_W / 2, y + 0.36, detail, fontsize=8.8, color=INK_2, ha="center", va="center", linespacing=1.35)
            centres[(r, i)] = (x, y)

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, color=MUTED, linewidth=1.3, shrinkA=0, shrinkB=0))

    for r, (_, direction, boxes) in enumerate(ROWS):
        for i in range(len(boxes) - 1):
            (x0, y0), (x1, _) = centres[(r, i)], centres[(r, i + 1)]
            mid = y0 + BOX_H / 2
            if direction > 0:
                arrow((x0 + BOX_W + 0.06, mid), (x1 - 0.06, mid))
            else:
                arrow((x0 - 0.06, mid), (x1 + BOX_W + 0.06, mid))
        if r + 1 < len(ROWS):  # last box of this row sits directly above the first box of the next
            (x0, y0), (_, y1) = centres[(r, len(boxes) - 1)], centres[(r + 1, 0)]
            arrow((x0 + BOX_W / 2, y0 - 0.06), (x0 + BOX_W / 2, y1 + BOX_H + 0.06))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
