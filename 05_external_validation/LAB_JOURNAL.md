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
