# When Does q-error Predict Plan Regret? Three Regimes of Cardinality-Estimation Error

*A short, reproducible note. Companion code: this repository.*

## Abstract

Cardinality-estimation (CE) research ranks estimators by **q-error**, yet it is well known that q-error is
an imperfect proxy for query-plan quality. We give a measurement-driven account of *when* it is a good
proxy and when it is not, and why. Treating plan selection as an argmin over a piecewise-linear cost
landscape, we show that plan **regret** (P-error) from CE error is governed by plan-cost geometry in a
**regime-dependent** way: (i) for *small* errors, a true-point **condition number** κ predicts regret and
out-predicts q-error; (ii) for *large* errors — where real learned estimators operate — an **average-case
sub-optimality** measure **ACS∞** predicts which queries are regret-prone, while q-error is near-useless at
the query level (Spearman ≈ 0.05); (iii) the *worst* case is Haritsa's MSO. The three are one cost-ratio
spectrum under three weightings. We prove a limit law ACS∞ = Σₖ rₖπₖ with cardinality-independent weights,
and validate everything on STATS-CEB and JOB-light with four released estimators under pre-registered
decision rules. The contribution is conceptual and empirical — an average-case companion to worst-case
robust query optimization — not a new estimator.

## 1. Setup

A query *q* has candidate plans P₁..P_K; plan k's internal (join) nodes are I_k; under cardinality vector
*c*, cost is C_k(c)=Σ_{S∈I_k} c_S (the C_out model). The optimizer picks k̂=argminₖ C_k(ĉ) under estimates
ĉ. **Regret** (P-error) is ρ(ĉ)=C_{k̂}(c)/C_{k*}(c) ≥ 1, with k* optimal under truth. Let rₖ=C_k(c)/C_{k*}(c)
be the true cost-ratios. q-error of a sub-plan is max(ĉ/c, c/ĉ). We model error as log ĉ_S = log c_S + ε_S,
ε_S ~ N(0,σ²) iid, sweeping σ from small (accurate) to large (inaccurate).

## 2. Small error: a condition number

Regret is a forward error; q-error is a backward error; the missing factor is a **condition number**. We
define κ via the closed-form distance to the nearest plan-switch boundary in log-cardinality space: for the
optimal plan P* and an alternative P′, the minimum L∞ log-perturbation that flips the choice is
δ = ½·ln(A/B), where A,B are the true-cardinality sums over the plans' non-shared internal nodes; κ = 1/min_{P′} δ.
**Result:** at small σ, ρ(per-query regret, κ) ≈ 0.7–0.8 (STATS-CEB 0.70, JOB-light 0.79), exceeding the
correlation with realized q-error, robust to query size (partial ρ ≈ 0.65), and **decaying to ~0 as σ
grows** — because the condition-number relation is local (Figure, left). Independently confirmed on
JOB-light.

## 3. Large error: average-case sub-optimality

At large σ the estimate is flung far across the cost-cell complex, so true-point quantities (κ, and — we
verify — cost-weighted or discriminative variants) stop predicting regret. We instead define
**ACS∞(q) = lim_{σ→∞} E_ε[ρ]**, the *average-case* analogue of Haritsa's worst-case MSO. **Result** on
STATS-CEB (real estimators inaccurate, median q-error 2–10⁵): ACS∞ predicts per-query regret at ρ ≈ 0.54,
versus **ρ ≈ 0.05 for q-error** and ρ ≈ 0.20 for κ (Figure, right). Pre-registered, margin-primary:
bootstrap 95% CI of (ρ_ACS − ρ_q) = [0.22, 0.74]; holds on held-out query halves and on an unseen
estimator (DuckDB-native); and is **regime-specific** — it does *not* win on small-error JOB-light. ACS∞
is estimator-independent: it answers "which queries is CE error dangerous for," a property of the query.

## 4. A limit law

**Theorem (informal).** As σ→∞, optimizer selection converges to a cardinality-free minimax-rank rule
k̂→argminₖ max_{S∈I_k} ε_S, and ACS∞(q) = Σₖ rₖ·πₖ, where rₖ are the true cost-ratios and πₖ are
**cardinality-independent** combinatorial selection probabilities of the plan set-system {I_k}. *Sketch:*
log C_k(ĉ) = LSE_{S∈I_k}(log c_S+ε_S) → max_{S∈I_k} ε_S as the ε-spread dominates the O(1) cost terms; the
argmin then depends only on subset ranks. We validate cardinality-independence numerically (corr of the
cardinality-aware and cardinality-free estimates → 0.994 as σ→30). ACS∞ thus **factorizes** into a
cardinality part (the cost-ratio spectrum) and a structure part (the combinatorial weights), and MSO = maxₖ rₖ
is the worst-case dual under the same spectrum.

## 5. Reconciliation, relation to prior work, limits

The "q-error vs plan-cost" debate is two **regimes** of one phenomenon; real estimators straddle the
boundary, so no single scalar error metric predicts their regret cleanly. This builds on plan diagrams /
POSP geometry (Reddy & Haritsa 2005), the q-error bound (Moerkotte–Neumann–Steidl 2009), MSO and robust
query processing (Haritsa), per-plan cost-integral robustness (Wolf et al. 2018), and the P-error /
Plan-Cost metric (Negi et al. 2021; STATS-CEB, Han/Zhu et al. 2022). The new pieces are ACS∞ (a *per-query*
average-case difficulty predictor over the optimizer's *choice*), its limit law, and the pre-registered
regime demonstration. **Limits:** C_out cost model; 123/146 STATS-CEB queries covered (monster-join
timeouts); ACS∞ is a query-level (not per-estimator) measure; the limit law is σ→∞ while real estimators sit
at finite σ. Corrections and missed prior art welcome.
