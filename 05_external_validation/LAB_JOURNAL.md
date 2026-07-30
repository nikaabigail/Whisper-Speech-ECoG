# External-validation lab journal

This file records implementation decisions, real-data integrity checks, and bugs
before confirmatory access. It is evidence for a later code-versus-protocol audit;
it is not a results section.

## 2026-07-27 — dataset selection and scope

- Selected SingleWordProductionDutch (SWPD, OSF `nrgx6`) because it supplies raw
  simultaneous audio/iEEG for 10 participants and supports representation
  reconstruction plus continuous-event work.
- Selected VocalMind v2 because it supplies original audio and 1000 Hz sEEG for
  repeated Mandarin words. The primary subset has 20 words × repetitions 1–5.
- Rejected a forced common classification endpoint: SWPD has 100 unique prompts per
  participant with no within-participant word repetition; VocalMind is trialized and
  cannot be presented as a free-running asynchronous benchmark.
- Historical Ivanova/Procenko weights remain out of scope. New subject-specific
  neural regressors and heads must be trained from scratch.

## 2026-07-27 — source integrity

### SWPD

- Downloaded the OSF archive to `C:\WhisperECoG\SWPD` outside Git.
- Exact archive bytes: `2,794,936,886`.
- SHA-256:
  `015bc9c565c3dbdc7259c01be54f62b3346cbd7dc5cec8156eb718f64b6cbcd9`.
- OSF HEAD reported an inconsistent smaller size; acceptance uses actual bytes plus
  the pinned SHA-256, not HEAD.
- `sub-01` inventory: iEEG `307511 × 127` at 1024 Hz; audio `14414532` samples,
  timestamp-derived rate `47999.187483...` Hz; 100 word events; 300.3 s.
- Code hard-locks `sub-02`–`sub-10` before path lookup.

### VocalMind

- Downloaded only original overt-word WAV and original vocalized-word sEEG to
  `C:\WhisperECoG\VocalMind` outside Git.
- Strict deep inventory passed for all 119 paired files: finite `3000 × 110` sEEG at
  1000 Hz and stereo 16-bit `132300 × 2` WAV at the released 44.1 kHz.
- Primary repetitions 1–5 contain 100 balanced trials. Repetition 6 has 19 trials
  (`ShuMu` is absent) and is reserved for development checks only.
- Dataset-index SHA-256:
  `37fb049168cd42124767a8c6503cfbbaca168e46d04dd462b9eefa8a470e1199`.
- Primary-split SHA-256:
  `3c0ad03e33799b065661f16a21467c44dbb30fe228b578d7d2e8bd6aa2559dcb`.

## 2026-07-27 — independent fidelity control

- Ran the exact modernized SWPD author-MEL pipeline on development subject `sub-01`:
  70–170 Hz envelope, 50 ms/10 ms frames, nine contexts from −200 to +200 ms,
  fold-train PCA-50, OLS to 23-bin MEL, sequential 10-fold CV.
- Obtained mean Pearson correlation `r = 0.519977...`, visually matching the
  approximately 0.52 `sub-01` bar in the publication figure.
- This was a pipeline smoke with 10 circular shifts. Its empirical `p=0.0909` is not
  a scientific null result and must not be reported as such. A full fidelity null
  requires the predeclared 1000 randomizations.
- No confirmatory SWPD participant was read.

## 2026-07-27 — clean environment validation

- Built a fresh local Python 3.10 environment from the locked requirements.
- Verified `torch 2.10.0+cu128`, CUDA runtime 12.8, RTX 5070 Laptop GPU (7.96 GiB),
  compute capability 12.0, and a finite forward/backward CUDA operation.
- Full test suite at this checkpoint: 59 passed.
- The common training engine includes finite-value/gradient guards, validation-only
  early stopping, atomic checkpoints, RNG/optimizer resume, and a test proving that
  interruption plus resume is identical to uninterrupted CPU training.

## 2026-07-27 — pre-launch bugs and protocol decisions

### SWPD absolute-versus-relative clock — critical, caught before real run

NWB stream timestamps start near `9062565.623` s, whereas events.tsv onsets are
recording-relative (`0...297` s). The first matched-linear implementation mixed the
two clocks. It would have produced invalid block bounds. The author-MEL fidelity path
reads complete synchronized streams and was unaffected. Required fix: one explicit
recording-relative event → absolute series-time conversion, stream-start consistency
checks, a synthetic large-offset regression test, and a real metadata/bounds smoke.

### Historical time-grid parity — high, resolved before training

Historical `02_whisper_sync` trained regression on the approximately 1000 Hz neural
timeline and then used `HIDDEN_STRIDE=10`, giving the word classifier a 100 Hz hidden
trajectory. A proposed 50 Hz external runner would have used 20× fewer regression
windows and changed Conv width 10 from 100 ms to 200 ms. Decision: reported runs use
exact 1000 Hz regression and 100 Hz hidden trajectories. A 50 Hz path may exist only
as a labelled engineering smoke and cannot enter an article table.

### Overlapping-window standardization cost — high, optimization required

Applying a channel standardizer separately to every overlapping 1001-sample window
repeats almost the same arithmetic roughly 1000 times. Required behavior is to fit
statistics on train trials only, transform each complete trial once (memory or
checksummed derived cache), and then slice immutable windows. Numerical equivalence
to per-window transformation must be covered by a test.

### VocalMind development-only timing audit

Only repetition 6 audio was inspected. A simple 20 ms RMS audit across 19 trials gave
onset min/median/max `0.66/1.04/1.30` s and offset `1.08/1.90/2.24` s. Thus lag-window
valid windows beginning at 1.0 s retain onset context for the released 3 s trials,
but very early/short utterances remain a disclosed boundary limitation. Padding is
not introduced. Repetitions 1–5 were not inspected for tuning.

### Downstream branch-count-matched ensemble control

Comparing three Whisper models only with one MEL model confounds representation
diversity and ensemble size. The production primary comparison is therefore fixed
L3+L4+L5 versus a three-initialization MEL probability ensemble. For outer seed `s`,
MEL replica seeds are `s`, `s+1000`, and `s+2000`; no best replica is selected.

### VocalMind author-code comparison

Official code was inspected at commit
`e1202bab23cc8a2c944e5e13264b2ce0a37b2d03` (Apache-2.0). Its released MEL target
uses 16 kHz audio, a 1024-sample (64 ms) FFT/window, 320-sample (20 ms) hop,
80 bins, 80–7600 Hz, and log10 amplitude. The primary branch-count-matched MEL control
uses these author spectral parameters followed by the same train-only PCA-50 as
Whisper. It also applies the same per-trial peak-absolute waveform normalization as
Whisper; because the official MEL path has no peak normalization, the matched
primary target is not called author-exact. The no-peak path remains a separate
fidelity/sensitivity extractor. A rep6-only polyphase-versus-official-librosa audit
gave correlation `0.999978`, mean absolute difference `0.00121`, and maximum
absolute difference `0.0849`; this is an extraction diagnostic, not a classifier
result. The generic 25 ms MEL is at most secondary.

The author EEG baseline uses CAR, HGA 70–150 Hz plus low-frequency <100 Hz,
downsampling to 200 Hz, and per-trial channel normalization before a CNN/3-layer
biGRU MEL decoder. Our external experiment intentionally applies the historical
Ossadtchi 10–200 Hz neural topology and train-only scaling identically to MEL and
Whisper. Therefore the author model is a dataset-fidelity reference, not an
architecture-identical comparator. This distinction must appear in the paper.

The official training loop uses `test_loader` when selecting the best checkpoint
and defines no independent validation partition. Our protocol deliberately assigns
a distinct validation repetition and keeps test behind an authorization gate. It is
therefore a stricter matched representation comparison, not a literal reproduction
of the authors' end-to-end evaluation.

### VocalMind pre-freeze quarantine — critical, resolved before training

A one-fold development run would numerically train/validate on repetitions that
later become held-out tests in other outer folds. The runner now rejects every
numeric `fast_smoke`/`pilot` invocation before creating an output directory or
loading a trial. Its development command is metadata/plan-only and reports
`numerical_training_allowed=false` and `test_gate_open=false`. Any pre-freeze
engineering forward/backward smoke is restricted to incomplete repetition 6 and
must not report a classification metric. After freeze, a bounded first epoch is
part of the immutable production run and resumes in a new output root bound by the
same config/commit; it is not a separate pilot.

## Gate state at end of entry

- SWPD confirmatory subjects: closed.
- VocalMind repetitions 1–5: indexed, no reported model training/evaluation started.
- VocalMind development plan fingerprint:
  `3d697e63c161f8170960f869f3acfe8da615d3c42f8310780be41a120899c4a9`;
  real-data PowerShell plan passed with 119/100/19 trials, 2000 regression windows
  and 200 hidden frames per trial, while the numeric-run and test-open flags stayed
  false.
- Complete external-validation suite after the VocalMind gate/amplitude/source-
  identity fixes: 87/87 passed; no heavy extraction or training was launched.
- Protocol: development draft, not frozen or tagged.
- Heavy CUDA training: not started from this directory.

## 2026-07-27 — VocalMind rep6 real CUDA engineering smoke

- Executed only `vocalized_word:DaNao:rep06`; no numeric file from repetitions
  1–5 was loaded and no classification performance metric was computed.
- Frozen Whisper-base revision loaded successfully. Shared-peak MEL and Whisper
  L3/L4/L5 target shapes were finite; one real ECoG-encoder forward/backward and
  one hidden word-head forward/backward both produced finite positive gradient
  norms.
- Runtime: `34.94 s`; PyTorch peak allocated CUDA memory: `214.15 MiB`.
  Diagnostic losses are shape/gradient checks only and are not scientific results.
- Host: RTX 5070 Laptop GPU, 7.96 GiB, compute capability 12.0, NVIDIA driver
  592.01, torch 2.10.0+cu128, CUDA runtime 12.8, cuDNN 91002.
- Receipt:
  `C:\WhisperECoG\smoke\vocalmind_rep6_smoke_20260727_v1\vocalmind_rep6_smoke_receipt.json`,
  SHA-256 `fd1b917f27d31adf8cb62c05dfc6f6c52f8b6660e49d75966f20e848a058481f`.
- Host preflight SHA-256:
  `203f62fbc4fb059cd8386eb83683664eb1294522cd2b3594cbf5e360be128d46`.
- Disk gate correctly rejected `D:` at 18.3 GiB free before any model work;
  the smoke was rerouted to `C:` with 70.8 GiB free. No existing data was moved.

## 2026-07-27 — SWPD sub-01 neural runner closure

- Fixed the absolute-versus-relative clock defect before matched-linear or
  full-neural real training. The adapter now maps recording-relative event
  seconds to each stream's absolute NWB clock explicitly; cached frame times
  remain recording-relative for later audio annotation.
- Real `sub-01` metadata/bounds smoke (no Whisper, no training) passed. iEEG
  start was `9062565.622919232`, audio start `9062565.622922681`; the difference
  was `3.4496 microseconds`. Common recording duration was
  `300.3037109375 s`.
- Nearest-sample half-open block bounds were contiguous for all five blocks:
  iEEG `[0,61505)`, `[61505,123007)`, `[123007,184507)`,
  `[184507,246008)`, `[246008,307511)`; corresponding audio bounds were also
  contiguous. Tiny reads at both ends of every interval succeeded.
- Production regression is fixed to the exact 1000 Hz grid. Each lag window is
  inclusive `[t-1000 ms,t]` (1001 samples), never padded, and never crosses a
  visual-block/split boundary. Fixed downstream hidden stride 10 yields the
  historical 100 Hz head trajectory. The 50 Hz mode is diagnostic-only.
- One representation-independent neural cache feeds MEL/L3/L4/L5: 127 usable
  channels, `CAR=false`, zero-phase notch 50/100/150 Hz, zero-phase 10–200 Hz
  band-pass, block-local polyphase 1024→1000 Hz, then a training-block-only
  channel standardizer applied once per whole block.
- Default three-initialization control units are MEL seeds `4,1004,2004` and Whisper
  L3/L4/L5 seed `4`; no best MEL replica selection is permitted. The fixed
  L345 probability ensemble remains future event-level work.
- Deterministic audio-energy candidates are saved as
  `audio_energy_candidate_unreviewed`. Visual cues are not an input to VAD.
  Event metrics and continuous heads remain blocked until a human-audited
  audio TSV and a checksum-bound approval receipt exist.
- SWPD-specific unittest suite: 22/22 passed. Complete external-validation
  suite: 84/84 passed. Python compilation, CLI parsing, and PowerShell syntax
  parsing passed. No heavy real extraction/training was launched.

## 2026-07-27 — frozen v1 and immutable VocalMind OOF closure

- Froze `vocalmind_primary_production.json` as `frozen_confirmatory`: all five
  cyclic folds, seeds `1,2,3,4,42`, three MEL initializations, fixed L3/L4/L5,
  exact temporal grid, preprocessing, selection rules, and test gate are now one
  versioned contract. A scientific-field change requires a new protocol version
  and output root.
- Production now requires a clean Git checkout, a complete source/dependency
  fingerprint, and a SHA-256-bound host preflight receipt before it can create the
  numeric run. A bounded one-epoch call resumes into the same immutable run; it
  does not create a second pilot surface.
- Added a fail-closed post-production OOF aggregator. It validates every one of the
  five held-out gates and all 150 validation-fixed/completion receipts before it
  reads any held-out result. For each training seed it then verifies and pools
  exactly 100 unique trials and recomputes accuracy, balanced accuracy, macro-F1,
  top-3 accuracy, per-class recall, and the predeclared `L345 - MELx3` contrast.
  Fold metrics are never averaged in place of the pooled 100-trial metric. The
  aggregation checkout/runtime must exactly match the production source identity.
- Five-seed intervals are explicitly descriptive optimization variability
  conditional on one participant (`biological n=1`), not population inference.
- Complete external-validation suite: 92/92 passed. Production-config validation,
  JSON parsing, Windows PowerShell parsing, OOF CLI, private-path/secret scan, and
  large/private-payload scan passed. Heavy SWPD/VocalMind production was not
  started by the release audit.

## 2026-07-28 — SWPD sub-01 seed4_v2 result freeze

- Completed the development-only full-neural regression run at
  `C:\WhisperECoG_Work\SWPD\runs\seed4_v2`. The immutable primary summary SHA-256
  is `2fab4111d4d868a3204a88768b42b8758271637aa8745a066cc01aa0ead10052`.
- The held-out block contains 58,062 frames. Training blocks were 1/2/3,
  validation block 0, and test block 4. No confirmatory participant was read.
- The production JSON reports the fixed three-initialization MEL control as
  `r=0.0167993` and standardized MSE `1.142776` in whitened PCA-50 coordinates.
  L3/L4/L5 PCA-coordinate correlations were approximately `0.01238`, `0.01905`,
  and `0.01510`.
- A read-only post-hoc diagnostic inverted the fixed train-only PCA transform and
  evaluated the saved predictions against untouched raw held-out MEL80 targets.
  The fixed MEL x3 mean correlation in original MEL coordinates was `r=0.46522`
  (Fisher mean `0.46603`, median component `0.47808`, 80/80 components positive).
  This is the appropriate directional comparison surface for the published raw
  acoustic-coordinate baseline, but it is not promoted to a preregistered primary
  result after test access.
- Exact forward-transform reconstruction agreed within approximately `5e-7`, and
  a -1000 to +1000 ms lag scan peaked at -50 ms (`r=0.469219`) versus zero-lag
  `r=0.465216`. This excludes a gross one-second alignment or sample-order defect;
  the small lag observation was not used to retune the held-out model.
- The exact modernized SWPD-author MEL/OLS development reference remains
  `r=0.5199771` under its different 10-fold/23-bin/high-gamma protocol. Its result
  JSON SHA-256 is
  `4f02fbedbceb59ac8a154b8482f439fc8ebf8cf3d7283455a50cbe0708e25d24`.
- Synchronous word classification, human-audited asynchronous event evaluation,
  and the fixed L3+L4+L5 event ensemble remain unevaluated and gated.
- Machine-readable frozen record:
  `results_records/swpd_sub01_development_seed4_v2_20260728.json`.

## 2026-07-28 — Podcast ECoG publication V2 closure

- Completed and validated all nine patients under protocol
  `paper_exact_batched_v2`; no requested patient is missing.
- Primary subject-level metric was fixed as held-out mean Pearson `r` in the
  `0–500 ms` response window. Author spectral 160D was `0.007950 ± 0.010820 SD`;
  Whisper L3/L4/L5 were `0.022994/0.022476/0.021244`; fixed L3+L4+L5 was
  `0.024381 ± 0.011489 SD` (`n=9`).
- Fixed L3+L4+L5 minus author spectral was `+0.016431 ± 0.011855 SD`, 95% t-CI
  `[+0.007318,+0.025543]`, wins `9/9`, paired two-sided `p=0.003174`.
- The train-only PCA160 dimensionality-matched ensemble also exceeded author
  spectral by `+0.014541`, 95% t-CI `[+0.004155,+0.024927]`, wins `9/9`,
  `p=0.012081`. Full 512D versus PCA160 was not significant (`p=0.110102`).
- The fixed ensemble did not exceed the best single Whisper layer on the peak
  surface (`4/9` wins, mean delta `-0.000203`, `p=0.706267`). The supported claim
  is therefore Whisper-versus-author representation improvement, not a proven
  ensemble-versus-best-layer improvement.
- V1 and V2 ensemble means were nearly identical (`0.024383` and `0.024381`). V2
  is the primary publication surface; V1 remains a reproducibility check.
- Source summary SHA-256:
  `b21c413c88a2ac4a318803b687523fbe98d6b3789f0539cc353aca7b80746a9f`.
- Machine-readable frozen record:
  `results_records/podcast_ecog_v2_20260728.json`.

## 2026-07-28 — SWPD source-paper re-read

- Re-read and visually inspected Verwoert et al., Scientific Data 9:434,
  `s41597-022-01542-9`. This is the source paper for the already-used SWPD data,
  not an additional external dataset.
- The official validation is 70–170 Hz Hilbert high-gamma, nine neural contexts
  from -200 to +200 ms, train-only neural PCA50, OLS to raw 23-bin log-MEL, and
  non-shuffled 10-fold frame CV. It is not word classification or asynchronous
  event decoding.
- The paper reports approximately `r=0.5–0.86` across patients in Fig. 4 and
  explicitly states that the score is driven mainly by speech-versus-silence.
- Our exact-modernized author baseline for `sub-01`, `r=0.5199771`, agrees with
  the approximately 0.52 first bar and closes the basic data/alignment
  reproduction check.
- Our neural `seed4_v2` uses a different architecture, target space and stricter
  block split; its PCA-space Whisper correlations must not be compared directly
  to the paper's raw-MEL correlation.

## 2026-07-28 — SWPD sub-01 frozen matched PCA50 comparison

- Preserved the exact-modernized author MEL23/OLS result as a separate
  reproducibility control (`r=0.5199771`); the runner verifies its immutable
  SHA-256 before fitting the matched experiment.
- Froze `configs/experiments/swpd_sub01_matched_pca50_v1.json`: targets
  MEL80/L3/L4/L5, shared frame IDs and five block folds, train-only standardized
  PCA50 whitening for neural and targets, identical OLS decoder, and one
  Fisher-pooled component-correlation metric. A protocol validator fails closed
  on all of these fields.
- Completed all five folds at
  `C:\WhisperECoG_Work\SWPD\runs\matched_pca50_sub01_v1`. Only `sub-01` was read;
  `sub-02…sub-10` remained code-locked.
- Aggregate all-frame test results, mean ± fold SD:
  MEL80 `r=0.01518±0.00589`, MSE `1.05106±0.05402`; L3
  `r=0.04283±0.01082`, MSE `0.89878±0.02322`; L4 `r=0.03446±0.00600`, MSE
  `0.91971±0.03179`; L5 `r=0.03691±0.00462`, MSE `0.87669±0.02880`.
- Fold-paired correlation deltas versus MEL80 were positive in every fold: L3
  `+0.02765±0.00635` (`5/5`), L4 `+0.01928±0.00267` (`5/5`), L5
  `+0.02173±0.00412` (`5/5`). These are descriptive temporal-fold results for
  one participant, not population inference.
- L3/L4/L5 PCA coordinates were not averaged into an ensemble because each
  train-only reducer defines a different basis. An ensemble requires a common
  downstream surface or prediction combination after inverse/defined mapping.
- Full suite: `96/96` passed. Independent artifact validation confirmed 14,515
  unique test rows, identical target test IDs, PCA50 train-only receipts, and
  finite arrays.
- Summary SHA-256:
  `c71e4f9c1ed999f80b5e158913e21aa5566be554ff4803256e18d5c9edde7b1e`.
- Machine-readable frozen record:
  `results_records/swpd_sub01_matched_pca50_v1_20260728.json`.

## 2026-07-28 — SWPD confirmatory matched PCA50 closure with data-QC exclusion

- The frozen all-subject queue completed `sub-01` through `sub-09`. The robust
  timestamp-rate estimator correctly recovered 1024 Hz for `sub-07`, whose first
  timestamp interval spans about five nominal samples; the old first-difference
  estimator had incorrectly reported 204.55 Hz.
- The official `sub-10` source recording is incomplete. Its events table contains
  100 word rows but only 95 positive-duration trials. The final five word rows have
  zero duration and all point to sample 291899, which is the final sample of the
  291900-sample iEEG recording. No missing neural/audio segment can be reconstructed.
- `sub-10` was therefore excluded in a post-access data-QC amendment. No values were
  imputed and no participant-specific 95-trial split was introduced. The original
  frozen protocol remains unchanged; the amendment is
  `configs/experiments/swpd_all_matched_pca50_v1_qc_amendment_sub10.json`.
- Primary population inference excludes development subject `sub-01` and uses
  `sub-02` through `sub-09` (`n=8`). The secondary analyzable cohort is
  `sub-01` through `sub-09` (`n=9`).
- Primary subject-level Fisher-pooled correlations (mean ± SD; 95% t-CI) were:
  MEL80 `0.02388 ± 0.00956` `[0.01588, 0.03187]`; L3
  `0.05327 ± 0.01828` `[0.03799, 0.06856]`; L4 `0.05397 ± 0.01726`
  `[0.03954, 0.06840]`; L5 `0.05522 ± 0.01685` `[0.04113, 0.06930]`.
- Whisper-minus-MEL80 deltas were positive in every confirmatory patient: L3
  `+0.02940 ± 0.00891`, L4 `+0.03009 ± 0.00793`, L5
  `+0.03134 ± 0.00743`, each `8/8` wins. Holm-adjusted paired-test p-values were
  `3.37e-5`, `2.67e-5`, and `1.98e-5`, respectively.
- This supports a matched Whisper-representation advantage over MEL80. It does not
  constitute an L3+L4+L5 ensemble result because the layer-specific train-only PCA
  coordinate systems are not aligned.
- Final external summary SHA-256:
  `5c6fa8cbcedaf81867e11f77aafd502779190e5fb3abc4aa6ab333bb6424f3f6`.
- Machine-readable frozen record:
  `results_records/swpd_matched_pca50_confirmatory_qc_v2_20260728.json`.
## 2026-07-29 — frozen contextual SWPD завершён

- После development-анализа только на `sub-01` зафиксирована система
  `Whisper L4 → train-only PCA50`; контроль — прямая регрессия в MEL80.
- Без смены системы завершены `sub-02…sub-09`; `sub-01` исключён из primary
  inference, `sub-10` исключён по ранее зафиксированной source-QC поправке.
- MEL80: `r=0,69187`, Whisper L4: `r=0,69290`, парная разница
  `+0,00103`, 95% t-CI `[+0,00002; +0,00204]`.
- Whisper выиграл у `6/8` пациентов; paired `t p=0,0462`, exact sign
  `p=0,2891`. Вывод зафиксирован как сопоставимость и малый пограничный прирост,
  а не существенное практическое превосходство.
- По нижним 20 MEL-бинам: `Δ=+0,00074`, 95% t-CI
  `[-0,00004; +0,00152]`, `p=0,0604`.
- Проверены SHA-256 всех восьми subject receipts, summary и prediction-файлов.
  Frozen run-contract fingerprint:
  `d6ebb1f5f18f2120d43b1d82fcaab8d00d62a4e2515497753fe31eab1c26df87`.

## 2026-07-30 — fixed-Q neural population завершён

- После выбора fixed-Q neural Whisper L4 только на development-пациенте
  `sub-01` система без нового подбора перенесена на `sub-02…sub-09`.
- Выполнены пять seeds (`1,2,3,4,42`) × пять временных folds; test открывался
  только после фиксации всех 200 model selections.
- Fixed-neural Whisper L4: `r=0,70935`, 95% t-CI `[0,62425; 0,79444]`.
  Согласованный линейный Whisper L4: `r=0,69290`; прямой MEL80-контроль:
  `r=0,69187`.
- Парная разница neural−linear L4: `+0,01645`, 95% t-CI
  `[−0,00240; +0,03529]`, paired `t p=0,0779`, exact sign `p=0,0703`;
  neural выиграл у `7/8` пациентов. Результат фиксируется как положительная
  тенденция, а не как статистически подтверждённое превосходство.
- Для навигационной фигуры приблизительно оцифрованы столбцы Figure 4a
  Verwoert et al. (2022). Их среднее для общей группы `sub-02…sub-09` равно
  `≈0,715`, наше — `0,709`. Это визуальный ориентир, а не прямой тест, поскольку
  опубликованный протокол использует MEL23/10-fold, а наш — MEL80/strict block5.
