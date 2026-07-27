# VocalMind primary overt-word runbook

Status: the v1 production config is `frozen_confirmatory`. Repetitions 1–5 may be
opened only by the production runner from a clean checkout of the exact
protocol-freeze commit. The separate development config remains metadata/plan-only;
repetition 6 remains the only permitted non-metric engineering smoke surface.

## Exact experiment

Only vocalized word repetitions 1–5 enter fitting or reported metrics. In fold
`k`, repetition `k` is held-out test, the next repetition cyclically is validation,
and the other three repetitions are training data (60/20/20 trials, all 20 words).
Repetition 6 is development-only for adapter, audio timing, shape, and memory
smokes; it is not a secondary performance set.

Every 3 s raw trial is handled as follows:

1. Use the release-provided 110 clean channels in their pinned CSV order at exact
   1000 Hz.
2. Apply one representation-independent fixed transform to the whole trial: CAR,
   zero-phase fifth-order high-pass 10 Hz, zero-phase notches 50/100/150 Hz with
   Q=25/100/150, and zero-phase fifth-order low-pass 200 Hz.
3. Fit one per-channel mean/scale artifact on training trials only. There is no
   per-trial z-score. Apply that same artifact to train, validation, and authorized
   test trials.
4. Use every valid inclusive lag window `[t-1000 ms, t]`: end samples
   1000…2999, 2000 windows per trial at 1000 Hz. No boundary padding is allowed.
5. Extract acoustic targets on the same 1 ms grid by local neighboring-frame
   interpolation. Fit StandardScaler/PCA-50 on training target rows only.
6. Train the common `OneSecondEcogEncoder`: 110 input channels → 30 learned spatial
   channels → depthwise temporal filtering/envelope stages → BiLSTM → 3030D hidden
   state → linear 50D acoustic target.
7. Retain hidden states with fixed stride 10, producing a 100 Hz, 200-frame trial
   trajectory. Feed 3030×200 into the common temporal Conv(100, width 10) →
   MaxPool(width 10) → BiLSTM(100 per direction) → 20-class softmax head.
8. Choose regression and word checkpoints only by minimum validation loss with
   early stopping. Closed-set argmax has no threshold to tune.

The four target branches use identical neural samples, preprocessing, architecture,
target dimension, optimizer policy, and splits:

- VocalMind-author spectral MEL80 parameters plus the same per-trial peak-absolute
  waveform normalization used by Whisper → train-only PCA50;
- frozen `openai/whisper-base` commit
  `e37978b90ca9030d5170a5c07aadb050351a65bb`, encoder L3/L4/L5, each →
  separate train-only PCA50.

The L3+L4+L5 result is the predeclared arithmetic mean of the three softmax
matrices. Production also requires three MEL initializations with deterministic
seeds `base`, `base+1000`, `base+2000`, averaged without subset search. The primary
contrast is L345 versus MEL×3; comparisons with single MEL and L4 are secondary.
The development plan declares one MEL initialization but does not run it. Any
engineering smoke before freeze is restricted to repetition 6 and may check only
adapter, target, shape, forward/backward, memory, and throughput behavior. It must
not produce or report a word-classification metric.

## Author MEL spectral parameters and deliberate differences

The MEL spectral port follows `src/audio_preproc.py:librosa_wav2spec` at official
repository commit `e1202bab23cc8a2c944e5e13264b2ce0a37b2d03`: mono 16 kHz audio,
STFT magnitude with `n_fft=win_length=1024` (64 ms), Hann window,
`hop_length=320` (20 ms), centered constant padding, librosa 80-bin MEL basis from
80–7600 Hz, and `log10(max(1e-6, mel_amplitude))`.

For the matched primary contrast, both MEL and Whisper first use identical
per-trial peak-absolute waveform normalization (epsilon `1e-8`). The official MEL
function does not do that, so the primary MEL is accurately described as
"author spectral parameters plus shared peak normalization", not author-exact.
The no-peak official-fidelity extractor remains available as a separate secondary
target sensitivity mode and does not add a production training unit. A rep6-only
extraction audit comparing our polyphase resampling with the official librosa path
found correlation `0.999978`, mean absolute difference `0.00121`, and maximum
absolute difference `0.0849`; these are target-extraction diagnostics, not
classification results.

This is a matched acoustic control, not a literal reproduction of the authors'
complete training procedure:

- their official code uses `test_loader` while choosing the best checkpoint and
  does not define an independent validation split;
- our protocol reserves a separate validation repetition and never uses held-out
  test results for epoch, model, seed, subset, or threshold selection;
- their neural preprocessing is CAR → HGA 70–150 Hz plus low-frequency <100 Hz →
  200 Hz and per-trial z-score;
- our controlled comparison intentionally keeps the historical project’s 10–200 Hz,
  1000 Hz neural topology and a train-only channel standardizer so MEL and Whisper
  differ only in acoustic target representation.

Therefore comparisons to the published VocalMind baseline must state these
differences and must not be described as an exact end-to-end reproduction.

## Test gate

For each fold, the test gate lists every configured outer seed and every predeclared
model initialization. In production that includes MEL×3 plus L3/L4/L5 for each
outer seed. Numeric test ECoG cannot be loaded and held-out evaluation cannot run
until every immutable validation-fixed receipt exists. The evaluator also checks a
cryptographic authorization token, split fingerprint, and exact ordered test IDs.

## Boundary limitation

A rep6-only development audio audit (no reps1–5 were used for tuning) found crude
20 ms RMS onset min/median/max 0.66/1.04/1.30 s and offset 1.08/1.90/2.24 s across
19 trials. Starting lag-window endpoints at 1.000 s preserves at least some onset
context, but the earliest word ends near 1.080 s, so it supplies few post-onset
frames. No padding or onset-dependent frame selection is introduced to compensate.

## Windows 11 development plan

From `05_external_validation` after bootstrap, the only permitted development
command is read-only planning:

```powershell
.\scripts\run_vocalmind_pilot.ps1 `
  -DataRoot "C:\WhisperECoG\VocalMind" `
  -OutputRoot "D:\WhisperECoG\runs\vocalmind_primary_pilot" `
  -PlanOnly
```

Without `-PlanOnly`, the development wrapper stops deliberately. Numeric production
uses `run_vocalmind_production.ps1`, a clean checkout of the announced freeze SHA,
and a brand-new `OutputRoot` named for that commit. A bounded one-epoch call must be
part of the same immutable production run and resumed from that same output root; it
is not a separate pilot result.

## Known production risks

- One participant limits inference to within-participant repeated-word replication.
- The 1000 Hz grid is compute-heavy: 120,000 training regression windows per fold
  before batching, per target/initialization.
- PCA-50 on 120,000×512 training rows is memory intensive; reducers and raw target
  caches should reside on a drive with ample free space.
- Trial-edge forward/backward zero-phase filtering and centered author STFT are
  explicitly offline preprocessing choices; they are not causal or real-time.
- Production refuses a dirty worktree, a non-frozen config, mismatched execution
  plan/preflight receipt, or source/dependency identity drift.

## Immutable five-fold OOF summary after production

Run this only after the production command has completed all five folds and has
written its final `summary.json`. The aggregator first validates every test-gate
receipt and every validation-fixed artifact across all folds. Only after all five
gates pass does it read the 25 seed/fold prediction artifacts. It verifies the
fixed class order, exactly 100 unique held-out trials per seed, probability hashes,
the predeclared L3/L4/L5 and MEL probability means, and the absence of test-driven
selection. Run it from the same clean freeze commit and locked runtime as training;
the aggregator compares its current source/runtime fingerprint with the immutable
production manifest and fails closed on any difference.

```powershell
.\scripts\aggregate_vocalmind_oof.ps1 `
  -RunRoot "D:\WhisperECoG\runs\vocalmind_frozen_v1"
```

The default output is the immutable `oof_aggregate` directory inside `RunRoot`.
It contains a JSON artifact with the aligned 100-trial probabilities and a tidy
CSV with closed-set metrics. Mean, SD, SEM, and 95% t intervals are descriptive
across the five training seeds conditional on this one participant. Neither folds
nor seeds are reported as independent biological samples. Re-running the command
only succeeds if the existing JSON and CSV are byte-identical.
