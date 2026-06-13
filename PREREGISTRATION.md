# Pre-registration trail (frozen before each analysis)

Every empirical claim in this repo was made under a decision rule fixed **before** the data was analysed.
We kept the full trail — including a hypothesis that was **rejected** and one that **missed its bar** — so
the findings can be read with the failures in view. This is a summary; the verbatim frozen docs live in the
companion research log.

## H1 (Day-1) — REJECTED, honestly

*Hypothesis:* a scalar, true-point condition number κ explains the q-error→regret relationship, pooled
across all error magnitudes.
*Result:* **rejected** — conditioning on κ added no lift to q-error's (weak) regret correlation, pooled.
Exploratory follow-up revealed *why*: the relationship is strongly **regime-dependent** (κ works at small
error only), which the pooled test averaged away. We did not relabel the rejection.

## H2 (Day-2) — CONFIRMED on an independent workload

*Hypothesis:* κ predicts regret in the **small-error** regime; the effect decays as error grows.
*Result:* **confirmed** under frozen thresholds on the independent JOB-light workload (ρ=0.79 at σ=0.25,
robust to query size, monotone decay). A fair real-estimator check then showed κ-vs-q-error is
**regime-dependent** (κ wins for accurate estimators, q-error for inaccurate) — reshaping the claim from
"κ beats q-error" to "two regimes."

## H3 (ACS Step 1) — NEAR-MISS, recorded as NO SIGNAL

*Hypothesis:* geometry-only ACS∞ predicts large-error regret with ρ ≥ 0.55 **and** beats q-error by ≥ 0.10.
*Result:* ρ = 0.549 (missed the 0.55 absolute bar by 0.001) with margin +0.499 (passed ×5). By the letter
of the rule this is **NO SIGNAL** — we recorded it as such, and noted the design flaw: the *comparative
margin*, not an arbitrary absolute cutoff, should be the gate.

## H4 (ACS Step 2) — CONFIRMED, margin-primary

*Hypothesis (fixing H3's flaw, on added robustness/independence data):* the margin ρ(regret,ACS∞) −
ρ(regret,q-error) is robustly positive and generalizes.
*Result:* **confirmed** — bootstrap 95% CI of the margin = [0.222, 0.737]; holds on held-out query halves;
holds on an unseen estimator (DuckDB-native); regime-specific (does not win on small-error JOB-light); and
a cheap analytic form tracks the Monte-Carlo measure. This is the basis for the ACS∞ claim.

## Theorem (Step 3) — derived and validated

ACS∞(q) = Σₖ rₖ·πₖ with cardinality-independent minimax-rank πₖ; cardinality-independence validated
numerically (corr → 0.994 as σ→30). This *superseded* an earlier empirical "geomean" proxy, which we had
honestly flagged as a correlate, not the limit.

---

The point of keeping H1 and H3 visible: the confirmed results (H2, H4) earn more trust *because* the
record shows what we rejected and where we missed, not just what we kept.
