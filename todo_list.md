# Training Pipeline — Remediation Plan

## Recommended Execution Order

- [x] **Phase 0 — Freeze overclaims and define the official baseline**
- [x] **Phase 1 — Repair temporal validity and leakage in `training.py`**
- [x] **Phase 2 — Repair statistical inference and benchmark comparisons**
- [x] **Phase 3 — Reduce feature-selection optimism**
- [x] **Phase 4 — Add rolling out-of-time validation**
- [x] **Phase 5 — Reassess population design if the use case is underwriting**

## Phase 0 — Freeze Overclaims

- [x] **Remove unsupported significance language from outputs and README**
- [x] **Treat `--reject-inference` as experimental, not part of the official benchmark**
- [x] **Treat stacking as experimental until temporal validity is restored**
- [x] **Clarify whether the target population is booked-only monitoring or underwriting-stage scoring**

## `training.py` Checklist

### 1. Temporal CV

- [x] **Refactor `TemporalExpandingCV` to split by time blocks, not row-count boundaries**
- [x] **Guarantee each fold satisfies `max(train_date) < min(val_date)`**
- [x] **Prevent the same date cohort from appearing in both train and validation**
- [x] **Add diagnostics for fold date ranges, row counts, and positive counts**
- [x] **Fail early when folds are too small for stable evaluation**

### 2. Reject Inference

- [x] **Restrict `compute_score_band_bad_rates()` to booked rows with `mis_Date < SPLIT_DATE`**
- [x] **Ensure no post-split matured outcomes are used to estimate pseudo-label rates**
- [x] **Keep `create_reject_pseudo_labels()` limited to pre-split rejects only**
- [x] **Add assertions that pseudo-label source statistics come from the allowed time window**
- [x] **Keep reject inference opt-in and excluded from the official summary tables**

### 3. Calibration

- [x] **Replace the random calibration split in `main()` with a temporal calibration block**
- [x] **Use the latest pre-split booked window for calibration and earlier data for fitting**
- [x] **Ensure calibration uses booked-only ground-truth rows even when reject inference is enabled**
- [x] **Log calibration date range, row count, and positive count**

### 4. Stacking

- [x] **Remove `StratifiedKFold(shuffle=True)` from official stacking**
- [x] **Either exclude stacking from official evaluation or rebuild it using temporal out-of-fold predictions**
- [x] **If rebuilding, train the meta-learner only on temporally valid out-of-fold base predictions**
- [x] **Keep stacking labeled experimental until the temporal implementation is complete**

### 5. Feature Discovery and Selection

- [x] **Separate interaction screening from final model estimation**
- [x] **Move `search_interactions()` and RFECV into a nested temporal workflow or a dedicated development window**
- [x] **Freeze the feature set before final out-of-time testing**
- [x] **Track feature provenance: raw, engineered, interaction, frequency, group-stat, RFECV-kept**
- [x] **Run ablations for raw features, engineered features, interaction search, RFECV, calibration, and reject inference**

### 6. Statistical Comparison Utilities

- [x] **Keep `evaluate()` for marginal per-model summaries**
- [x] **Add paired bootstrap delta functions for AUC, PR AUC, and Brier**
- [x] **Add DeLong ROC AUC comparison or another defensible paired AUC test**
- [x] **Write a new comparison output comparing each candidate model to `risk_score_rf` and `score_RF`**
- [x] **Stop using overlap of marginal confidence intervals as the decision rule for model comparisons**

### 7. Output and Control Flow

- [x] **Make development, calibration, and test populations explicit in `main()` variable naming**
- [x] **Separate official outputs from experimental outputs**
- [x] **Exclude experimental modes from the primary leaderboard CSVs**
- [x] **Update logging so every training stage reports its date window and sample definition**

### 8. Population Design (Underwriting)

- [x] **Add explicit `population_mode` handling for booked monitoring vs underwriting**
- [x] **Load booked plus rejected/canceled decisioned applications for underwriting runs**
- [x] **Keep booked-only evaluation explicitly labeled as a proxy population in underwriting mode**
- [x] **Tag benchmark, rolling OOT, and ablation outputs with `population_mode` and `evaluation_population`**
- [x] **Write `population_summary.csv` and `applicant_scores_post_split.csv` for underwriting runs**

## Test Suite Checklist

### 1. Fix Stale Tests First

- [ ] **Update `tests/test_features.py` to match current `engineer_features()` column names**
- [x] **Update all callers to the current `temporal_split()` return signature**
- [x] **Update `tests/test_pipeline.py` to the current `train_stacking()` signature**
- [ ] **Remove or rewrite assertions that reflect outdated methodology**

### 2. Temporal Split and CV Tests

- [x] **Add tests that each temporal fold satisfies strict past-to-future ordering**
- [x] **Add tests that date cohorts are not split across train and validation**
- [x] **Add tests that training folds expand monotonically over time**
- [x] **Add tests that tiny time windows fail clearly**

### 3. Reject Inference Tests

- [x] **Add tests that `compute_score_band_bad_rates()` excludes rows on or after `SPLIT_DATE`**
- [x] **Add a regression test where post-split outcomes would change band bad rates if leakage existed**
- [x] **Add tests that pseudo-labeled rejects are always pre-split**
- [x] **Add tests that rows without `risk_score_rf` are excluded from pseudo-labeling**
- [x] **Add tests that booked-only rows are the sole source of band bad-rate estimation**

### 4. Calibration Tests

- [x] **Add tests that the calibration block is later than the fit block**
- [ ] **Add tests that calibration remains pre-test**
- [x] **Add tests that calibration uses only booked ground-truth rows**
- [ ] **Add tests for calibration metadata and date-range logging helpers if introduced**

### 5. Stacking Tests

- [x] **If stacking is removed from official runs, add tests that it is excluded by default**
- [x] **If temporal stacking is implemented, add tests that each row receives exactly one temporal out-of-fold prediction**
- [x] **Add regression tests proving the meta-learner never trains on future-informed predictions**

### 6. Statistical Comparison Tests

- [x] **Create `tests/test_statistics.py`**
- [x] **Test paired bootstrap deltas on identical score arrays**
- [x] **Test paired bootstrap deltas on clearly better score arrays**
- [x] **Test comparison output schema and column names**
- [x] **If DeLong is implemented, add sanity tests for identical and clearly distinct predictors**

### 7. End-to-End Pipeline Tests

- [x] **Refocus `tests/test_pipeline.py` on the official non-experimental path**
- [x] **Add an official-baseline smoke test with temporal calibration and no reject inference**
- [x] **Add an experimental-mode smoke test only if reject inference or stacking remains available**
- [x] **Assert artifact presence by mode so official and experimental outputs remain separated**

### 8. Underwriting Population Tests

- [x] **Add targeted tests for `build_population_summary_df()`**
- [x] **Add targeted tests for `build_applicant_score_frame()`**
- [x] **Add artifact assertions for underwriting output CSVs**
- [x] **Run a broader regression subset around the underwriting refactor**

## File-Level Breakdown

### `training.py`

- [ ] **Edit `TemporalExpandingCV`**
- [ ] **Edit `compute_score_band_bad_rates()`**
- [ ] **Edit `create_reject_pseudo_labels()` guardrails**
- [ ] **Edit calibration split logic in `main()`**
- [x] **Edit `train_stacking()` or remove it from the official flow**
- [x] **Add paired comparison utilities**
- [x] **Update artifact writing for paired comparison outputs**

### `tests/conftest.py`

- [x] **Fix `temporal_split()` fixture unpacking**
- [x] **Add fixtures with explicit time structure**
- [ ] **Add fixtures that expose leakage if post-split rows are mistakenly used**

### `tests/test_features.py`

- [ ] **Update stale expected engineered feature names**
- [ ] **Keep assertions aligned with the current implementation**

### `tests/test_reject_inference.py`

- [x] **Add leakage-specific tests**
- [x] **Add time-boundary and source-population tests**

### `tests/test_split_and_eval.py`

- [x] **Update `temporal_split()` unpacking**
- [x] **Add stricter temporal invariants**
- [ ] **Keep metric smoke tests**

### `tests/test_pipeline.py`

- [x] **Fix outdated `train_stacking()` usage**
- [x] **Refocus on official pipeline behavior**
- [x] **Add mode-aware artifact assertions**

### `tests/test_statistics.py`

- [x] **Create the file**
- [x] **Add paired bootstrap delta tests**
- [x] **Add comparison schema tests**
- [x] **Add inferential edge-case tests**

## Acceptance Criteria

- [ ] **No official training path uses future labels**
- [ ] **All validation, calibration, and ensemble construction are temporally defensible**
- [ ] **The test suite matches the actual current API**
- [ ] **Benchmark claims are backed by paired model-comparison outputs**
- [x] **Experimental modes are clearly separated from official results**
