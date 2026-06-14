"""Figure: predicting REAL PostgreSQL runtime regret. Values from the k=3, full-coverage
(110/111 queries) run on a dedicated PostgreSQL 13.1 with injected estimator cardinalities
(costmodel/run_pg.py + analyze_pg.py). margin(ACS_inf - q-error) bootstrap 95% CI = [0.34, 0.82].
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

names = ["q-error", r"$\kappa$ (local)", r"ACS$_\infty$"]
vals = [-0.158, -0.020, 0.417]
fig, ax = plt.subplots(figsize=(5.4, 4.0))
bars = ax.bar(names, vals, color=["#bbb", "#88a", "#3a6"])
ax.axhline(0, color="grey", lw=0.8)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + (0.02 if v >= 0 else -0.05),
            f"{v:+.2f}", ha="center", fontsize=11)
ax.set_ylabel(r"$\rho$( real PG runtime regret , predictor )")
ax.set_title("Predicting real PostgreSQL runtime regret\n(k=3, 110/111 queries; injected estimator cardinalities)")
ax.set_ylim(-0.3, 0.55)
fig.tight_layout()
fig.savefig("runtime_pg.pdf"); fig.savefig("runtime_pg.png", dpi=150)
print("wrote runtime_pg.pdf + .png")
