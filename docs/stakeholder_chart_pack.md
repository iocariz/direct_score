# Stakeholder Results Chart Pack

## Purpose

This document is a stakeholder-facing outline for presenting the current underwriting-mode model results in a concise, visual format.

It is designed for leadership, credit-risk stakeholders, and underwriting partners who want to understand:

- what changed in the pipeline
- what population is now covered
- how the best candidate model compares with current benchmarks
- what the main caveats are before any policy change

---

## Audience

- Business leadership
- Credit risk / underwriting stakeholders
- Model governance reviewers
- Product or operations partners involved in decisioning

---

## Regenerate the charts

```bash
uv run python stakeholder_charts.py --output-dir output
```

Generated PNG assets are written to `output/plots/`.

---

## Suggested Presentation Structure

### Slide 1. Executive headline

**Title:** Underwriting-stage scoring now covers the full decisioned population

**Visual:** Large KPI tiles

![Slide 1 - Underwriting KPI tiles](../output/plots/stakeholder_kpis.png)

**Show:**

- Post-split decisioned applications scored: **1,180,746**
- Post-split booked applications: **50,735**
- Post-split rejected applications: **980,972**
- Post-split canceled applications: **149,039**
- Held-out booked proxy test size: **16,109**
- Held-out observed defaults: **611**

**Key takeaway:**

The pipeline has moved from a booked-account monitoring setup to an underwriting-stage scoring setup that produces scores for the full post-split decisioned population.

**Speaker note:**

This is the main business change. Even though the measurable repayment outcome still exists only for booked accounts, the system now produces applicant-stage scores for all decisioned cases after the split date.

---

### Slide 2. What changed

**Title:** From booked-only monitoring to underwriting-stage scoring

**Visual:** Before/after process diagram

![Slide 2 - Before and after underwriting refactor](../output/plots/stakeholder_before_after.png)

**Left side: Before**

- Focused on booked accounts only
- Produced monitoring-style benchmark comparisons
- Did not persist a full applicant-stage scoring output

**Right side: Now**

- Supports explicit population modes
- `underwriting` mode scores `Booked`, `Rejected`, and `Canceled` applications
- Saves `population_summary.csv`
- Saves `applicant_scores_post_split.csv`
- Keeps official evaluation explicitly labeled as `booked_proxy`

**Key takeaway:**

The pipeline now supports operational underwriting analysis, not just booked-loan monitoring.

---

### Slide 3. Population coverage

**Title:** Population coverage expanded to the full decisioned funnel

**Visual:** Stacked bar chart for pre-split vs post-split population by status

**Use data from:** `output/population_summary.csv`

![Slide 3 - Population coverage](../output/plots/stakeholder_population_coverage.png)

**Suggested values to plot:**

| Split | Booked | Rejected | Canceled | Total |
|-------|--------|----------|----------|-------|
| Pre-split | 51,380 | 974,536 | 180,023 | 1,205,939 |
| Post-split | 50,735 | 980,972 | 149,039 | 1,180,746 |

**Key takeaway:**

Most of the decisioned population is non-booked, so a stakeholder view of underwriting performance must consider the broader funnel, not only booked accounts.

**Speaker note:**

The model now scores the entire decisioned population post-split, which makes it suitable for threshold analysis, routing analysis, and future policy simulation.

---

### Slide 4. Hold-out benchmark comparison

**Title:** Best candidate model improves on `score_RF`, but not on `risk_score_rf`

**Visual:** Horizontal bar chart of held-out ROC AUC and PR AUC

**Use data from:** `output/results.csv`

![Slide 4 - Hold-out benchmark comparison](../output/plots/stakeholder_holdout_benchmarks.png)

**Suggested values to show:**

| Model | ROC AUC | PR AUC |
|-------|---------|--------|
| risk_score_rf (benchmark) | 0.6613 | 0.0879 |
| Logistic Regression | 0.6552 | 0.0641 |
| XGBoost | 0.6403 | 0.0595 |
| CatBoost | 0.6347 | 0.0577 |
| LightGBM | 0.6319 | 0.0569 |
| score_RF (benchmark) | 0.6185 | 0.0538 |

**Key takeaway:**

The strongest current in-house candidate is **Logistic Regression** on the held-out booked proxy test. It clearly outperforms `score_RF`, but `risk_score_rf` remains the strongest benchmark overall.

**Speaker note:**

The takeaway for stakeholders is not that we have already beaten every benchmark. The stronger message is that the new pipeline materially improves on one internal benchmark and extends scoring coverage to the full decisioned population.

---

### Slide 5. ROC and precision-recall curves

**Title:** The relative ranking is visible across the full threshold range

**Visual:** Side-by-side ROC curve and precision-recall curve on the official held-out booked proxy sample

**Use data from:**

- `output/holdout_test_scores.csv`
- `output/results.csv`

![Slide 5 - ROC and precision-recall curves](../output/plots/stakeholder_roc_pr_curves.png)

**Key takeaway:**

The curve view shows the same headline pattern across thresholds: `risk_score_rf` remains the strongest benchmark overall, while **Logistic Regression** is the strongest current in-house candidate.

**Speaker note:**

This chart is useful when stakeholders ask whether the bar-chart ranking is driven by one arbitrary threshold. The answer is no: the relative ordering is visible across the full operating range.

---

### Slide 6. Statistical significance vs current benchmarks

**Title:** Logistic Regression shows a significant lift over `score_RF`

**Visual:** Forest plot of paired AUC improvement with confidence intervals

**Use data from:** `output/benchmark_comparisons.csv`

![Slide 5 - Paired AUC lift](../output/plots/stakeholder_auc_lift.png)

**Suggested comparisons to plot:**

| Candidate | Reference | AUC Improvement | 95% Interval | DeLong p-value |
|-----------|-----------|-----------------|--------------|----------------|
| Logistic Regression | score_RF (benchmark) | +0.0367 | [+0.0190, +0.0555] | 0.000219 |
| XGBoost | score_RF (benchmark) | +0.0218 | [+0.0034, +0.0413] | 0.025961 |
| CatBoost | score_RF (benchmark) | +0.0162 | [-0.0012, +0.0347] | 0.087523 |
| LightGBM | score_RF (benchmark) | +0.0134 | [-0.0062, +0.0333] | 0.198038 |
| Logistic Regression | risk_score_rf (benchmark) | -0.0060 | [-0.0347, +0.0232] | 0.692702 |

**Key takeaway:**

There is strong statistical evidence that Logistic Regression improves over `score_RF`, but there is no evidence that the current candidate outperforms `risk_score_rf`.

**Speaker note:**

This is a better stakeholder message than quoting overlapping confidence intervals. It directly answers whether the observed uplift versus a benchmark is likely to be real.

---

### Slide 7. Rolling out-of-time performance

**Title:** Performance is reasonably consistent across forward validation windows

**Visual:** Line chart by fold or grouped bar chart of mean ROC AUC and mean PR AUC

**Use data from:**

- `output/rolling_oot_summary.csv`
- `output/rolling_oot_results.csv`

![Slide 6 - Rolling out-of-time performance](../output/plots/stakeholder_rolling_oot.png)

**Suggested summary values:**

| Model | Mean ROC AUC | Mean PR AUC | Folds |
|-------|--------------|-------------|-------|
| risk_score_rf (benchmark) | 0.6587 | 0.0841 | 4 |
| XGBoost | 0.6380 | 0.0606 | 4 |
| CatBoost | 0.6344 | 0.0639 | 4 |
| Logistic Regression | 0.6339 | 0.0576 | 4 |
| score_RF (benchmark) | 0.6264 | 0.0544 | 4 |
| LightGBM | 0.6064 | 0.0508 | 4 |

**Key takeaway:**

The model ranking is directionally stable over time, with `risk_score_rf` still strongest and the in-house candidates consistently ahead of `score_RF` on average.

**Speaker note:**

This helps address the common stakeholder question: “Was the held-out test just lucky?” The rolling OOT view shows the conclusions are not driven by one isolated time window.

---

### Slide 8. Calibration message

**Title:** Calibration improves probability quality for learned models

**Visual:** Small side-by-side bar chart of raw vs calibrated Brier score

**Use data from:** `output/results.csv`

![Slide 7 - Calibration comparison](../output/plots/stakeholder_calibration.png)

**Suggested values:**

| Model | Raw Brier | Calibrated Brier |
|-------|-----------|------------------|
| Logistic Regression | 0.2248 | 0.0361 |
| XGBoost | 0.2233 | 0.0362 |
| CatBoost | 0.2208 | 0.0363 |
| LightGBM | 0.0364 | 0.0363 |

**Key takeaway:**

Calibration materially improves the probability estimates, which matters for policy thresholds, expected-loss analysis, and decisioning cutoffs.

**Speaker note:**

For stakeholders, the important message is not the technical definition of Brier score. It is that calibrated probabilities are more usable for downstream policy decisions.

---

### Slide 9. Recommended stakeholder conclusion

**Title:** What we can say today

**Visual:** Three callout boxes

**Callout 1: What is already true**

- The pipeline now supports underwriting-stage scoring across the full decisioned population
- The best current candidate materially improves on `score_RF`
- Results are stable enough to justify deeper policy analysis

**Callout 2: What is not yet proven**

- The current candidate does not outperform `risk_score_rf`
- Headline evaluation is still based on booked accounts only
- Reject inference remains experimental and is not part of the official result set

**Callout 3: What should happen next**

- Run threshold and approval-rate simulations on `applicant_scores_post_split.csv`
- Compare routing strategies against current benchmarks
- Quantify expected-loss or profitability trade-offs before policy deployment

**Key takeaway:**

This work is ready for stakeholder review as an underwriting analytics capability, but not yet ready to claim a universal benchmark replacement.

---

## Recommended Chart Order for a Deck

1. Executive headline
2. What changed
3. Population coverage
4. Hold-out benchmark comparison
5. Statistical lift vs benchmarks
6. Rolling out-of-time performance
7. Calibration message
8. Recommendation and next steps

---

## Plain-English Caveats to Keep in the Deck

- The model now scores all decisioned applications, but observed repayment outcomes still exist only for booked accounts.
- That means current headline performance is a **booked-proxy** estimate of underwriting quality, not a full causal measurement of applicant-level grant/decline performance.
- `risk_score_rf` remains the strongest benchmark in this run.
- Reject inference and stacking should stay labeled **experimental**.

---

## Data Sources Behind This Pack

- `output/results.csv`
- `output/benchmark_comparisons.csv`
- `output/rolling_oot_summary.csv`
- `output/rolling_oot_results.csv`
- `output/population_summary.csv`
- `output/applicant_scores_post_split.csv`

---

## Recommended Follow-up Deliverables

After this chart pack, the next most useful documents would be:

1. **Executive summary** for leadership approval
2. **Policy simulation memo** using score cutoffs on `applicant_scores_post_split.csv`
3. **Technical appendix** with methodology, definitions, and validation caveats
