# SWPD matched PCA50 confirmatory code and artifact audit

Date: 2026-07-28
Run ID: `matched_pca50_all_v2` (external run directory is not published)
Outcome: **PASS WITH DOCUMENTED LIMITATIONS**

## Executive conclusion

No critical correctness defect was found in the frozen SWPD confirmatory result.
The saved result is reproducible from the checksummed block caches, train-only
reducers and linear models. The matched comparison supports the following claim:

> Under identical temporal splits, train-only PCA50 transforms, OLS decoder and
> held-out metric, each tested Whisper layer is more predictable from ECoG than
> MEL80 across all eight confirmatory patients.

The run does **not** establish an L3+L4+L5 ensemble, word-classification accuracy,
speech-only performance, causal decoding or asynchronous decoding.

## Audit scope and evidence

The audit independently verified:

- the frozen protocol, dataset manifest and six source-code hashes recorded before
  the run;
- all five cached blocks for each analyzable participant;
- all train, validation and test sample-ID hashes and their disjointness;
- one shared train-only neural PCA50 reducer per fold;
- separate train-only PCA50 target reducers for MEL80, L3, L4 and L5;
- all saved OLS coefficients by independently refitting every model;
- all saved test predictions and fold metrics;
- patient-level aggregation, confidence intervals, paired tests and Holm correction;
- exclusion of development subject `sub-01` from primary inference;
- the raw source-file defect underlying the `sub-10` QC exclusion;
- absence of a trained model or test result for `sub-10`.

Independent artifact totals:

| Checked item | Count |
|---|---:|
| Analyzable participants | 9 |
| Primary confirmatory participants | 8 |
| Temporal folds | 45 |
| Train-only reducers | 225 |
| Independently refitted OLS models | 180 |
| Saved prediction rows across all targets | 522,620 |

The machine-readable receipt is
`05_external_validation/swpd_matched_pca50/audit_receipt.json`.

## Protocol-to-code correspondence

| Protocol requirement | Code evidence | Audit result |
|---|---|---|
| Five adjacent temporal blocks | `make_visual_blocks()` in `matched_linear.py` | Pass |
| One held-out test block per fold | `run_matched_folds()`, split construction around line 451 | Pass |
| Neural transform fitted only on training frames | reducer call around line 463; immutable train-ID receipt | Pass |
| Targets standardized and reduced to PCA50 using training frames only | target reducer call around line 495 | Pass |
| Same neural input and OLS decoder for all targets | shared `train_x`; `LinearRegression` around line 507 | Pass |
| Same held-out IDs for MEL80/L3/L4/L5 | checked in every saved prediction file | Pass |
| Patient is the population unit | one fold-aggregated value per patient | Pass |
| Three predeclared Whisper-minus-MEL contrasts | `aggregate_subject_summaries()` | Pass |
| Holm correction across three tests | `swpd_matched_all.py`, step-down adjustment around line 245 | Pass |
| Development subject excluded | primary cohort is `sub-02` through `sub-09` | Pass |

## Statistical reproduction

Primary system correlations (`n=8`, mean ± patient SD):

| System | Correlation | 95% t-CI |
|---|---:|---:|
| MEL80 | 0.02388 ± 0.00956 | [0.01588, 0.03187] |
| Whisper L3 | 0.05327 ± 0.01828 | [0.03799, 0.06856] |
| Whisper L4 | 0.05397 ± 0.01726 | [0.03954, 0.06840] |
| Whisper L5 | 0.05522 ± 0.01685 | [0.04113, 0.06930] |

Paired effects versus MEL80:

| Contrast | Mean Δr ± SD | Wins | Holm-adjusted p |
|---|---:|---:|---:|
| L3 − MEL80 | +0.02940 ± 0.00891 | 8/8 | 3.37e-5 |
| L4 − MEL80 | +0.03009 ± 0.00793 | 8/8 | 2.67e-5 |
| L5 − MEL80 | +0.03134 ± 0.00743 | 8/8 | 1.98e-5 |

As a non-preregistered robustness check only, 8/8 effects in the same direction
give an exact two-sided sign-test p-value of `0.0078125` for each layer.

## Findings and limitations

### 1. Validation block is not used for model selection — moderate

Each fold reserves three blocks for training, one for validation and one for test.
OLS has no selected hyperparameter, so the validation block is evaluated but does
not affect the fitted model. This is not leakage and is identical across systems;
it is conservative because only 60% of frames train each fold. A separate
four-train-block/one-test-block sensitivity analysis could improve data efficiency,
but it must not replace the frozen primary result.

### 2. Primary metric is all-frame representation predictability — moderate

The outcome is a continuous acoustic-representation correlation. It is not word
accuracy. Because there is no independently audited speech mask, speech-versus-
silence structure may contribute. The paper must describe this as representation
decoding rather than discrete speech decoding.

### 3. Feature extraction is offline and noncausal — moderate

High-gamma extraction uses zero-phase filtering, and Whisper encoder targets are
bidirectional within each audio chunk. This is appropriate for the present offline
comparison but cannot support a real-time/asynchronous claim.

### 4. `sub-10` exclusion was decided after data access — moderate

The exclusion is objective and independent of model performance: the final five
word rows have zero duration at the last recorded sample. The audit confirmed that
no `sub-10` model or test result exists. The exclusion, amendment date and resulting
primary `n=8` must remain explicit in the manuscript and cohort diagram.

### 5. Confirmatory sample is small — minor

Population inference uses eight patients. The consistency of all 8/8 paired effects
is encouraging, but effect sizes, confidence intervals and patient-level points
should be emphasized more than p-values.

### 6. No mathematically defined layer ensemble in this experiment — claim boundary

Each target has its own train-only PCA basis. L3, L4 and L5 PCA coordinates therefore
cannot be averaged directly. A future ensemble needs a shared downstream output
space or a separately preregistered train-only fusion transform.

## Decision before GitHub publication

The existing primary result does not need to be rerun. The following should be
published together:

1. the original frozen protocol;
2. the dated `sub-10` QC amendment;
3. the final JSON/CSV checksums and compact result record;
4. this audit report and machine-readable receipt;
5. the publication figures and captions;
6. an explicit statement that the result is offline matched representation decoding.
