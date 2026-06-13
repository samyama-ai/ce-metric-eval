"""Conceptual: the small-error condition number as distance to the nearest plan-switch
boundary. Two plans' costs cross at delta = 1/2 ln(A/B); kappa = 1/delta. Output: flip_margin.pdf/png.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

A, B, C0 = 1000.0, 100.0, 500.0          # non-shared sums (A from P', B from P*) + shared baseline
delta = 0.5 * np.log(A / B)               # closed-form flip margin
u = np.linspace(-1.5, 3.0, 400)
cost_star = B * np.exp(u) + C0            # P* gets costlier as we perturb toward the wall
cost_alt = A * np.exp(-u) + C0            # P' gets cheaper

fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.plot(u, cost_star, lw=2, color="#2a7", label=r"cost of $P^\ast$ (truth-optimal)")
ax.plot(u, cost_alt, lw=2, color="#36c", label=r"cost of alternative $P'$")
ax.axvline(delta, color="#c33", ls="--", lw=1.3)
ax.axvspan(-1.5, delta, color="#2a7", alpha=0.06)
ymax = float(cost_alt.max())
ax.annotate(r"$\delta=\frac{1}{2}\ln(A/B)$" + "\n(nearest wall)", (delta, ymax * 0.62),
            xytext=(delta + 0.25, ymax * 0.72), color="#c33", fontsize=10,
            arrowprops=dict(arrowstyle="->", color="#c33"))
ax.text(-1.3, ymax * 0.18, r"$P^\ast$ stays optimal" + "\n(small error)", color="#176", fontsize=9.5)
ax.text(delta + 0.55, ymax * 0.18, r"plan flips to $P'$", color="#236", fontsize=9.5)
ax.set_xlabel(r"log-perturbation $u$ toward the boundary  ($u=\log q\text{-error}$)")
ax.set_ylabel(r"plan cost $C_{\mathrm{out}}$")
ax.set_title(r"Condition number $\kappa = 1/\delta$: distance to the nearest plan-switch boundary")
ax.legend(loc="upper center", fontsize=9, framealpha=0.9)
ax.set_xlim(-1.5, 3.0)
fig.tight_layout()
fig.savefig("flip_margin.pdf"); fig.savefig("flip_margin.png", dpi=150)
print(f"wrote flip_margin.pdf + .png  (delta={delta:.3f})")
