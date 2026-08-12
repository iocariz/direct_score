# Codebase Audit — direct_score

**Date:** 2026-08-12
**Scope:** All pipeline modules (`training*.py`, `scoring.py`, `model_governance.py`, `stakeholder_charts.py`, `generate_report.py`, `main.py`), test suite, project configuration, and repo hygiene. ~10,500 lines of Python reviewed; test suite executed live.
**Method:** Five parallel in-depth reviews (pipeline orchestration; feature engineering / temporal machinery; models / reporting / constants; scoring / governance / report generation; tests / hygiene), followed by independent verification of every critical and high finding against the source.

---

## Executive summary

The core modelling discipline is **sound**: all five CLAUDE.md invariants (SCRPLUST1 exclusion, target maturity, temporal split, booked-only training, no shuffled CV) hold in the default path, the statistics layer (BCa bootstrap, DeLong, Holm-Bonferroni, monthly block resampling) is implemented correctly, and the 310-test suite is genuinely assertive and passes clean.

However, the audit found **2 critical and 7 high-severity defects**:

1. A one-line bug (`max(value, fallback)` with `fallback=1000`) **silently disables early stopping for every boosted model** — almost certainly the root cause of the documented "tree models still overfit" limitation.
2. The fitted feature-engineering state (frequency maps, group medians, cardinality caps) is **never persisted**, so the production scoring contract cannot be fulfilled — guaranteed train/serve skew.
3. `ScoringService` loads the **uncalibrated** model while applying calibrated PD tier cuts.
4. The applicant-population fair-lending analysis is **unreachable dead code** in the default mode.
5. `generate_report.py` prints **hardcoded conclusions** ("statistically significant", "rules out data leakage") not derived from the data, plus an already-stale hyperparameter table.

Severity scale: **Critical** = corrupts the deployed artifact or its headline claims; **High** = materially wrong output in a production or governance surface; **Medium** = biased diagnostics, latent crash, or maintainability trap; **Low** = minor/cosmetic.

---

## 1. CLAUDE.md rule compliance (verified)

| Rule | Status | Evidence |
|---|---|---|
| SCRPLUST1 excluded from all modelling | ✅ Holds | Only in `DROP_COLS` (`training_constants.py:140`); absent from `RAW_NUM`/`RAW_CAT` (sole sources of interaction candidates), `MISS_CANDIDATES`, `GROUP_STAT_PAIRS`, `REJECT_SCORE_COL`. Guarded by `tests/test_features.py:62-72`. |
| 12-month target maturity (`mis_Date <= 2025-01`) | ✅ Holds | `enforce_matured_target` hard-filters (`training_features.py:48-86`); interaction search applies the same filter. |
| Temporal split only | ✅ Holds | `training.py:331-332`; no `train_test_split` anywhere in source. |
| Booked-only supervised training | ✅ Holds by default | `training.py:263, 288`. Exception: opt-in `--reject-inference` (see M-13). |
| No `StratifiedKFold(shuffle=True)` | ✅ Holds literally | Zero occurrences. Spirit-of-the-rule caveats: H-5 (TargetEncoder internal CV) and the `CalibratedClassifierCV(FrozenEstimator)` internal split (leakage-safe, calibration holdout only). |

Also verified clean: all randomness seeded (`RANDOM_STATE=42` throughout, including Optuna, SHAP subsampling, reject downsampling); the model-selection window / final untouched window separation (`training.py:349-375`) is genuine; calibration holdout is temporally last, booked-only, and never used for calibration-method selection; benchmark score negation is consistent across evaluation surfaces.

**Stale doc:** CLAUDE.md says "Stacking is experimental (non-temporal OOF predictions)" — this is no longer true. `training_stacking.py:225-270` uses `TemporalExpandingCV`, refits fresh base pipelines per fold on strictly earlier dates, and excludes never-predicted rows. A dedicated anti-leakage test asserts every fold's max train date < min validation date (`tests/test_pipeline.py:448-467`). Update CLAUDE.md, not the code.

---

## 2. Critical findings

### C-1. Early stopping is silently disabled for all boosted models
`training_models.py:85-88`, exploited at `:360, :369, :386-389` (LGBM), `:477-481, :489, :506-509` (XGB), `:594-598, :606, :623-626` (CatBoost).

```python
def normalize_estimator_count(value, fallback: int = 1) -> int:
    if value is None or pd.isna(value):
        return fallback
    return max(int(value), fallback)
...
fold_best_iters.append(normalize_estimator_count(clf.best_iteration_, fallback=N_ESTIMATORS_CEILING))
```

With `fallback=N_ESTIMATORS_CEILING` (1000), `max(int(value), 1000)` clamps every real per-fold `best_iteration_` (necessarily ≤ 1000) **up** to 1000. `select_conservative_boosting_rounds` therefore receives `[1000, 1000, ...]`, its 25th-quantile logic is a no-op, and every final booster is refit at the full 1000-tree ceiling with no early stopping. The `fallback` was clearly intended only for the `None`/NaN branch (the default `fallback=1` makes `max()` a floor-at-1).

**Consequence:** the Optuna objective scores early-stopped models, but the deployed artifacts are trained with ~1000 rounds — CV estimates do not describe the model that ships. This is the most likely root cause of the documented limitation "Tree models still overfit significantly vs Logistic Regression despite tightened bounds". No test covers these functions.

**Fix:** in the three `fold_best_iters.append(...)` call sites (and the three final-fit `normalize_estimator_count` calls), keep `fallback=N_ESTIMATORS_CEILING` for the missing-value branch but do not pass it as a floor — e.g. split the function into `value if value is not None else fallback` plus a separate floor-at-1.

### C-2. Feature-engineering recipes are never persisted — the scoring contract is unfulfillable
`scoring.py:291-292` vs `training_features.py:604-621, 642, 663`.

The scoring docstring instructs callers to "Run engineer_features + add_interactions + add_modeling_features (or use the saved feature_engineering artifacts) before scoring", but **no such artifacts are saved**. The fitted state is train-frame-dependent and discarded:
- frequency encodings: `freq = X_train[col].value_counts(normalize=True)` (`:642`);
- group-stat medians: `X_train.groupby(cat)[num].median()` (`:663`);
- cardinality caps: `cardinality_maps[col] = set(top_cats)` (`:609`) — returned and thrown away at every call site (`X_train_out, X_other_out, _ = reduce_cardinality(...)`).

Only interaction bin edges survive (via `interaction_leaderboard.csv`). At serve time a consumer cannot compute `FREQ_*`, `*_VS_*`, or the "Other" bucket for an applicant without the original training DataFrame — every consumer must hand-roll an approximation (e.g. batch-local frequencies), silently shifting feature values and mispricing PDs. The 50 %-missing guard (`scoring.py:263`) still permits scoring with all engineered columns NaN-imputed.

**Fix:** persist a single fitted feature-engineering transformer (or JSON of the maps) alongside the models, and have `ScoringService` apply it.

---

## 3. High findings

### H-1. `ScoringService` loads the uncalibrated model, then applies calibrated PD tier bands
`scoring.py:151-165`. The recommended base name from `model_selection.csv` resolves to e.g. `logistic_regression.joblib`; `*_calibrated.joblib` is only a fallback **if the base file is missing** — and training always saves both (`training_reporting.py:871-874`). Governance states the calibrated variant is the production candidate (`training_reporting.py:1296-1298`), yet raw scores are bucketed against absolute PD cuts `(0.03, 0.06, 0.10, 0.20)`. For isotonic-needing tree models the assigned risk tiers are systematically wrong.

### H-2. Applicant-population adverse-impact (fair-lending) analysis is unreachable dead code
`training.py:2452-2456` gates on `"AGE_T1" in applicant_scores_df.columns`, but `build_applicant_score_frame` deliberately omits `AGE_T1` (PII-minimisation comment at `training.py:1362-1370`; the frame carries only `mis_Date`, `status_name`, target, benchmarks + scores). Underwriting is the default mode, so the intended `analysis_population = "underwriting_applicants"` AIR analysis can never run and the pipeline silently degrades to booked-holdout-only analysis — rejection-threshold disparate impact is measured only on already-accepted accounts. A material governance regression for a regulated PD model.

### H-3. Hardcoded conclusions in the validation report
`generate_report.py`:
- `:218` asserts the model "significantly outperforms score_RF" with no p-value check; `:227-231` asserts non-significance and "risk_score_rf retains a meaningful lead" without checking the actual p-value or the sign of the improvement.
- `:780-781` prints "No feature exceeds IV 0.30, which rules out data leakage" unconditionally — never compared to `iv["iv"].max()`.
- `:394-399` canned ablation conclusion never derived from `ablation_results.csv`.

A validation report that can lie is the most dangerous failure mode for a regulator-facing artifact. Every prose conclusion should be conditional on the number it cites.

### H-4. Report's hyperparameter table is already factually wrong
`generate_report.py:412-414` states LGBM `num_leaves 8-31`, XGB `max_depth 2-4, min_child_weight 20-100`, CatBoost `depth 3-5, min_data_in_leaf 50-300`; actual bounds in `training_models.py` are `num_leaves 8–63` (`:313`), `max_depth 3–7, min_child_weight 20–200` (`:440-441`), `depth 3–7, min_data_in_leaf 50–500` (`:554-556`). Generate these rows from the code/constants, not prose.

### H-5. TargetEncoder's internal cross-fitting is non-temporal (within-train look-ahead)
`training_features.py:824-827` (also `:880`), used by the LR/EBM preprocessor built at `training.py:558`:

```python
# Keep TargetEncoder deterministic and non-shuffled to preserve temporal discipline.
("encoder", TargetEncoder(smooth=target_encoder_smooth, cv=5, shuffle=False)),
```

`shuffle=False` does not make the cross-fitting temporal: fold *k* is still encoded on all other folds, including future ones, and after reject augmentation (`training.py:521-527`, `ignore_index=True`) row order is not even date-sorted, so the folds aren't date-contiguous either. The comment overstates what `shuffle=False` buys. Not test leakage (fit only on the development window), but early rows are encoded with later rows' targets, optimistically biasing encoded features, Optuna fold scores, and stability selection. Fix: a custom temporal cross-fitting encoder, or accept and document the bias.

### H-6. Reject-inference pseudo-labels embed calibration-window outcomes
`training_reject_inference.py:48-53`: score-band bad rates are computed on all pre-split booked rows (`< SPLIT_DATE`), but `training.py` computes them (step 4) **before** `temporal_calibration_split` (`:568`) reserves the last ~15 % of dates. Pseudo-labels attached to fit-window rows therefore carry aggregated target information from the calibration holdout, compromising its independence (test discipline is intact). Fix: compute band rates only on the eventual fit window.

### H-7. CatBoost double-rebalances classes under reject inference
`training_models.py:534-590, 634-647`: LGBM and XGB guard with `effective_pos_weight = 1.0 if sample_weight is not None` (`:299`, `:426`), but CatBoost unconditionally sets `auto_class_weights="Balanced"` while also passing `sample_weight=w_fold`, and its `pos_weight` parameter is dead. In reject-inference mode CatBoost probabilities are biased toward defaults, making its calibration metrics and cross-model comparison unfair.

### H-8. Engineering hygiene: broken documented test command, no CI, no lockfile
- `uv sync && uv run pytest` (per CLAUDE.md/README) **fails on a clean checkout** with `ModuleNotFoundError: No module named 'numpy'` — pytest lives in `[project.optional-dependencies] test` (`pyproject.toml:24-25`), so the working command is the undocumented `uv sync --extra test && uv run pytest`. Move pytest to a `[dependency-groups] dev` group or fix the docs.
- **No CI** (`.github/workflows/` absent): the 9-minute, 310-test suite never runs automatically.
- **No lockfile was committed** despite `.gitignore:100-104` saying it should be; 14 dependencies are unpinned `>=` floors, so environments are not reproducible — a real problem for a regulated model. *(A `uv.lock` resolved during this audit has been committed on this branch as a starting point.)*

---

## 4. Medium findings

- **M-1. "Reject top X %" off-by-one** — `training_reporting.py:1186-1189`: threshold taken at descending index `cutoff_idx` then applied with `>=`, so ≥ `cutoff_idx + 1` accounts are rejected (more with ties). All precision/recall/FPR/capture figures in the business threshold table sit at a slightly wrong operating point. Correct: index `cutoff_idx - 1` with `>=` (or `cutoff_idx` with `>`).
- **M-2. Same off-by-one in the adverse-impact analysis** — `training_reporting.py:1679-1680`: feeds the per-age-band approval rates and the 0.80-rule PASS/FAIL flags.
- **M-3. Post-hoc ensemble double-dips the calibration holdout** — `training_reporting.py:1798-1844`: blend partner and weight are grid-searched by PR AUC on `y_calibration`, the same data used to fit the calibrators — contradicting `training_constants.py:107-110`'s own design note. Test metrics stay honest; the calibrated ensemble variant is optimistically fit.
- **M-4. Frequency/group stats computed over the calibration window** — `add_modeling_features` is called (`training.py:541-547`) on full pre-split development before the calibration split (`:568`), so fit-window rows get `FREQ_*`/`*_VS_*` values derived from future data. The later re-train paths (`training.py:1751-1756`, `:1839-1844`) do it correctly with fit-only stats. Unsupervised stats only; test properly excluded.
- **M-5. `pd.cut` crash on duplicate quantile edges in reject banding** — `training_reject_inference.py:58-61` (reused `:113`): no dedup of `np.quantile` edges; empirically confirmed `ValueError: Bin edges must be unique` on a score with mass points (contrast `_fit_binned_numeric_labels`, `training_features.py:308`, which dedupes).
- **M-6. Reject pseudo-rows double-counted in encoding statistics** — each reject becomes two full rows (bad+good) *before* `add_modeling_features`, so `value_counts`/`median` count each reject twice at weight 1.0 — 4× the intended `REJECT_SAMPLE_WEIGHT = 0.5` influence on `FREQ_*`/`*_VS_*`.
- **M-7. Ratio interactions scored on a different definition than deployed** — search uses `va / vb` → `inf` (rows excluded from selection AUC), deployment uses `df[a] / df[b].replace(0, np.nan)` → median-imputed (`training_features.py:399-400` vs `:555-556`). Selection evidence excludes exactly the rows where production behaviour differs.
- **M-8. Rolling OOT validation overlaps HPO data** — `training.py:812-827`: rolling windows re-use rows that served as Optuna CV validation folds; hyperparameters were selected partly against them, so the "stability" criterion (20 % of the selection score) is optimistically biased. Final test window unaffected.
- **M-9. `MISS_` flag existence decided on full data including test** — `training_features.py:155-160` via `training.py:464-468`: `df[col].isna().mean()` over pre+post-split rows (decision-level leakage). Also recomputed independently on the rejected frame: a column straddling the 1 % threshold between populations raises `KeyError` in `augment_training_data` (`training_reject_inference.py:177`) or silently NaN-fills applicant scores (`training.py:1337`).
- **M-10. Final pipelines share mutable preprocessor instances** — EBM and XGB embed the same `prepared["preprocessor"]` object; LGBM and CatBoost share `prepared["lgbm_preprocessor"]` (`training.py:702-744`; `training_models.py:263-264, 517-518, 634-635`). sklearn `Pipeline.fit` does not clone steps — correct today only because all fit on identical data; any refit of one silently corrupts its siblings.
- **M-11. Serve-time handling of rare-but-seen categories diverges from training** — training maps rare levels to `"Other"` before encoding (`training_features.py:611`); scoring has no `cardinality_maps` (see C-2), so such categories hit `TargetEncoder`'s unseen path (global mean) / `OrdinalEncoder(unknown_value=-1)` instead of the trained "Other" encoding — a quiet PD shift on sparse segments.
- **M-12. `variable_dictionary.csv` provenance is wrong for every feature** — `model_governance.py:575` reads `prov.get("source", ...)` but `build_feature_provenance` writes the column as `"provenance"` (`training_features.py:799`), so the fallback always fires and interactions/frequency/group-stat features are all mislabeled "engineered". (The model card handles this correctly via `_feature_source_column`.)
- **M-13. `--reject-inference` contaminates diagnostics** — WoE/IV (`training.py:1076-1080`) and step-16/17 data-quality outputs consume pseudo-labeled rows unweighted, so `iv_summary.csv` and the data-quality report include synthetic labels when the flag is on. (Calibration/evaluation are correctly restricted to booked rows via `w == 1.0` masks.)
- **M-14. Duplicated metric/selection code across reporting surfaces can diverge** — three variants of `_sanitize_output_name` (`training_reporting.py:27`, `stakeholder_charts.py:86-87`, `scoring.py:72-92`); fallback "selected model" sort order differs between `stakeholder_charts.py:128` and `generate_report.py:57`; capture-at-10 % recomputed by hand with different tie handling (`stakeholder_charts.py:810-818` vs `training_reporting.py:1186-1189`); PSI cuts 0.10/0.25 hardcoded in two places instead of using the constants.
- **M-15. `n_pos` from an arbitrary benchmark row presented as the holdout default count** — `generate_report.py:140-141` and `stakeholder_charts.py:186` take row 0 of `benchmark_comparisons.csv` (a pair-specific, finite-score subset count); charts render **0** observed defaults when no comparison row exists.
- **M-16. Committed binaries and notebook outputs bloat history** — `report.docx` (2.4 MB), `methodology_report.docx` (2.0 MB), `executive_brief.docx` (215 KB) — all regenerable outputs of `scripts/build_*.py`; `notebooks/1-eda.ipynb` (4.6 MB with 23 embedded PNGs); a tracked `.DS_Store` (also missing from `.gitignore`); three NPE SAS macros unrelated to the pipeline at repo root, of which `npe_macro_kk.sas` is a renamed scratch copy of ambiguous canonicity.
- **M-17. Stale documentation** — README and CLAUDE.md claim "239 tests" (actual: 310, all passing); README's project-structure section omits 4 `training_*` modules, `generate_report.py`, `scripts/`, `docs/`; README embeds images from the gitignored `output/plots/` (broken on fresh clone); `run_summary.md` is a stale committed artifact referencing gitignored files; no lint/type-check tooling configured anywhere.

---

## 5. Low findings (condensed)

- **L-1.** HPO per-fold score computed on the same fold used for early stopping (`training_models.py:347-359` et al.) — mildly optimistic, currently moot given C-1.
- **L-2.** `_lgbm_prauc_eval` double-applies a sigmoid (`training_models.py:116-119`) — rank-invariant, logged metric value only.
- **L-3.** Inconsistent "is a probability" thresholds (`training_reporting.py:1036` `<= 1.01` vs `:267` `<= 1.0 + 1e-9`).
- **L-4.** `select_best_model` calibration score can go far below 0 when a candidate has NaN Brier (fallback 1.0 outside the min-max range, no clamp; `training_reporting.py:1336-1347`).
- **L-5.** Two uncoordinated "overfit" thresholds: flag at delta > 0.03 (`training_constants.py:172`) vs penalty ramp 0.01–0.10 with hardcoded magic numbers (`training_reporting.py:1387`).
- **L-6.** Optuna pruner warmup comment off by one (first prune after 4 folds, not 3; conservative direction).
- **L-7.** `extract_feature_importance` name alignment via non-strict `zip` and assumed `num_cols + cat_cols` order (`training_reporting.py:660-687`) — silent truncation if the transformer changes column count.
- **L-8.** SHAP computed twice per run, second call overwriting artifacts (`training.py:1072-1073`, `:2401`); redundant `except (ValueError, TypeError, Exception)` (`:1957`).
- **L-9.** `compute_psi` doesn't dedupe percentile edges (unlike `compute_csi`) — degenerate bins understate PSI on tie-heavy scores (`training.py:2112-2116`).
- **L-10.** Dead code from the stage refactor: ~30 unpacked-but-unused variables in `main()`, `models` assigned twice (`training.py:2619, 2632`), discarded `_run_diagnostics_and_governance` return (`:1096`); `%%` renders literally in loguru messages (`:2487, 2497, 2500`); step numbering jumps 4 → 7.
- **L-11.** `reduce_cardinality` maps NaN → "Other", conflating missingness with rare levels (`training_features.py:611-612`); consistent train/test, information-loss only.
- **L-12.** `checksums.json` absence is non-fatal and it lives next to the artifacts it checks — integrity, not tamper, protection (`scoring.py:185-190`). Path traversal itself is well defended.
- **L-13.** No cross-artifact consistency check at load: a partially regenerated output dir (new model, stale dictionary) loads without complaint; `score_batch` performs no OOD validation (single-applicant path only) (`scoring.py:192-219, 335-359`).
- **L-14.** Non-temporal LOO target-encoding fallback (`training_features.py:188-199`) — only reachable with < 3 date blocks; `add_interactions` bin-edge refit fallback can touch test rows if metadata is lost (`:579`).
- **L-15.** Model-card / report narrative says "highest weighted score" but the selector's tiebreak band can pick a non-top-scoring model (`training_reporting.py:1476-1494` vs `model_governance.py:260-265`, `generate_report.py:504`); report claims table "sorted by PR AUC" but emits file order (`generate_report.py:461-471`); hardcoded benchmark names and unadjusted p-values in `create_auc_lift_chart` (`stakeholder_charts.py:356, 370`); gains chart assumes exactly 10 deciles (`:598`); crash-prone unguarded accesses in `generate_report.py` (`:194, :197, :799`); `pyproject.toml` placeholder description and global `FutureWarning`/`UserWarning` suppression; no regression test enforcing the shuffle ban (unlike SCRPLUST1, which has one); notebooks numbered 1- and 3- with no 2-.

---

## 6. Test suite assessment

**Executed live:** `uv sync --extra test && uv run pytest -q` → **310 passed, 0 failed** in ~9 min. No `xfail`/`skip` marks; only 4 conditional data-sufficiency skips, none triggered.

Strengths: high assert density (e.g. 56 asserts across 17 governance tests), property-style statistics tests (DeLong, bootstrap), schema-faithful seeded synthetic fixtures in `conftest.py` (no dependence on the real parquet), and explicit leakage guards (SCRPLUST1, temporal split direction, temporal CV ordering, stacking OOF anti-leakage).

| Module | Coverage |
|---|---|
| training.py | Good (end-to-end smoke + split/eval tests; `cli()` untested) |
| training_features.py | Good |
| training_reporting.py | Good |
| training_models.py | Indirect but adequate — **except the estimator-count helpers (C-1), which have zero coverage** |
| training_temporal.py / training_stacking.py / training_reject_inference.py | Good (incl. dedicated OOF anti-leakage test) |
| model_governance.py | Good |
| scoring.py | Adequate |
| stakeholder_charts.py | Thin (4 tests / 49 KB) |
| generate_report.py | Thin (1 test / 50 KB) |
| scripts/ (2,072 lines) | None |

---

## 7. Prioritized recommendations

1. **Fix C-1** (estimator-count clamp) and add a regression test asserting `normalize_estimator_count(500, fallback=1000) == 500`. Then re-run training — tree-model overfitting should drop materially.
2. **Fix C-2 + H-1 + M-11**: persist the fitted feature-engineering state as an artifact, make `ScoringService` apply it, and load the calibrated winner by default.
3. **Fix H-2**: carry `AGE_T1` (or a pre-binned age band) through `build_applicant_score_frame` so the applicant-population AIR analysis actually runs; fix the M-1/M-2 cutoff off-by-one it depends on.
4. **Fix H-3/H-4**: make every conclusion in `generate_report.py` conditional on the value it cites; generate the hyperparameter table from `training_models.py` constants.
5. **Fix H-6/H-7/M-5/M-6** before relying on `--reject-inference` results.
6. **Hygiene**: add a GitHub Actions workflow (`uv sync --extra test && uv run pytest`), keep `uv.lock` committed (added on this branch), fix the documented test command, update stale counts/structure in README + CLAUDE.md (incl. the stacking note), remove `.DS_Store` + regenerable `.docx` binaries (or move to LFS/releases), strip notebook outputs, adopt ruff.
7. **Medium-term**: temporal cross-fitting for `TargetEncoder` (H-5); move `add_modeling_features` after the calibration split (M-4); clone preprocessors per pipeline (M-10); deduplicate the tripled sanitize/selection/metric logic (M-14).
