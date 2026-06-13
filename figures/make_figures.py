"""Regenerate the headline figure from the analysis summaries.

Usage: python figures/make_figures.py [results_dir]   (default: data/results)
Reads confirm_summary.json (kappa regime curve) and acs_summary.json (large-error trio).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/results")
OUT = Path(__file__).resolve().parent


def main():
    confirm = json.loads((RES / "confirm_summary.json").read_text())
    acs = json.loads((RES / "acs_summary.json").read_text())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: kappa regime decay
    for label, d in confirm.items():
        if not isinstance(d, dict) or "curve" not in d:
            continue
        cur = d["curve"]
        xs = sorted(float(s) for s in cur)
        ys = [cur[str(s) if str(s) in cur else f"{s}"]["rho_kf"] for s in xs]
        ax1.plot(xs, ys, marker="o", label=label.split(" (")[0])
    ax1.axhline(0, color="grey", lw=0.7, ls=":")
    ax1.set_xlabel("estimation error scale  σ  (lognormal)")
    ax1.set_ylabel("ρ( plan regret , condition number κ )")
    ax1.set_title("κ predicts regret — only in the small-error regime")
    ax1.legend(fontsize=9)

    # Panel B: large-error predictors (STATS-CEB)
    s = acs["STATS-CEB (large-error)"]
    names = ["q-error", "κ (true-point)", "ACS∞ (this work)"]
    vals = [s["rho_regret_qmean"], s["rho_regret_kappa"], s["rho_regret_acs"]]
    bars = ax2.bar(names, vals, color=["#bbb", "#88a", "#3a6"])
    ax2.set_ylabel("ρ( query regret , predictor )")
    ax2.set_title("Large-error regime (STATS-CEB): what predicts query regret")
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    ax2.set_ylim(0, max(vals) * 1.25)

    fig.tight_layout()
    fig.savefig(OUT / "regimes.png", dpi=140)
    print(f"wrote {OUT/'regimes.png'}")


if __name__ == "__main__":
    main()
