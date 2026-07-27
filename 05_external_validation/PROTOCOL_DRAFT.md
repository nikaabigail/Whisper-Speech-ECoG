# External-validation protocol — frozen confirmatory v1

**Status:** frozen for the first confirmatory VocalMind run by the Git commit that
contains this file and `vocalmind_primary_production.json`. Production is permitted
only from a clean checkout of that exact commit. Any change to a scientific field,
split, metric, seed, model revision, or test gate creates a new protocol version and
requires a new output root. The filename is retained to preserve existing links.

## 1. Claim under test

A patient-specific invasive-neural decoder trained against frozen Whisper-base
encoder representations (L3/L4/L5), followed by a fixed equal-weight L3+L4+L5
probability ensemble, generalizes beyond the original private Russian ECoG data and
outperforms a matched MEL-target pipeline.

The historical Ivanova/Procenko weights are never transferred. Every neural decoder
and downstream head is trained from scratch for the new subject and split. Only the
frozen acoustic target extractor and predeclared architecture/hyperparameters carry
over.

## 2. Common controlled comparison

All representations receive identical neural samples, split IDs, context, optimizer
policy, early-stopping rule, and downstream model.

- frozen multilingual `openai/whisper-base` at revision
  `e37978b90ca9030d5170a5c07aadb050351a65bb`;
- encoder hidden states L3, L4, L5 extracted in one forward pass;
- fixed one-second neural context `[t-1000 ms, t]`;
- regression is trained on the exact 1000 Hz neural timeline, with acoustic
  targets aligned locally to every 1 ms frame; hidden trajectories are then
  sampled with the historical stride 10, giving a 100 Hz word-head timeline;
- a 50 Hz runner is permitted only as a labelled engineering smoke test and
  cannot contribute a reported scientific result;
- target transforms fitted on train only and serialized with the checkpoint;
- matched 50-dimensional targets for the full neural architecture;
- MEL, L3, L4, and L5 trained separately;
- fixed equal-weight mean of L3/L4/L5 softmax probabilities;
- primary contrast: L345 versus a three-initialization MEL probability ensemble
  matching the number of separately trained downstream branches (not exact total
  FLOPs); secondary contrasts use single MEL and predeclared L4;
- for outer seed `s`, MEL replica seeds are fixed as `s`, `s+1000`, and `s+2000`;
  no replica is selected or discarded by validation/test performance;
- no layer, ensemble subset, smoothing, threshold, or epoch is selected on test.

The exact author-MEL reproductions remain separate fidelity experiments because their
features, temporal context, and evaluation protocols differ from our architecture.

## 3. SWPD: development and confirmatory separation

`sub-01` is development-only: NWB adapter checks, exact 23-bin MEL reproduction,
speech-boundary annotation audit, and one-seed smoke training.

`sub-02` through `sub-10` remain closed until the protocol is frozen. Within each
subject, 100 trials are assigned to ten sequential blocks and then to five adjacent
20-trial pairs. The inexpensive linear representation analysis uses rotating
out-of-fold predictions. The full neural L3/L4/L5 experiment uses one fixed
60/20/20 split per patient so the five-seed experiment remains computationally
tractable: for `sub-N`, test-pair index is `(N-2) mod 5`, validation is the next
pair cyclically, and the other three pairs are train. Thus test position is spread
across patients without inspecting any signal or label outcome. Padding never
crosses a split boundary and edges at least as wide as the full receptive field are
discarded.

The exact author baseline is reproduced first:

- high gamma 70–170 Hz, 100/150 Hz suppression;
- 50 ms window, 10 ms step;
- temporal context -200 to +200 ms in 50 ms steps;
- train-only standardization and neural PCA-50;
- OLS to 23-bin log-MEL;
- sequential 10-fold CV and circular-shift null.

The leakage-controlled matched representation experiment then uses one common 20 ms
grid and train-only StandardScaler/PCA-50 for both MEL-80 and each 512D Whisper
target, with an identical decoder. It reports
Fisher-z averaged component correlation, standardized MSE, explained variance, and
speech-only versions of those metrics. Representation-specific correlations are not
reported as a literal percentage improvement without qualification.

For the full neural system, independently derived acoustic speech intervals create a
binary continuous head per layer. Visual cue intervals are never treated as speech
ground truth. VAD/energy labels are manually audited without access to model
predictions. The model receives neither cue phase nor time since stimulus.

Primary continuous metrics are event PR-AUC, validation-selected event F1, precision,
recall, FP per minute of true silence, and onset latency median/IQR. Any operating
point near recall 0.40 is selected on validation and applied unchanged to test.

Required controls include circular temporal shift, cue-phase-only baseline,
speech-only reconstruction, latency/pre-onset analysis, and acoustic-contamination
checks. The periodic experimental schedule is disclosed as a limitation.

## 4. VocalMind

The primary analysis uses only vocalized word trials with original synchronized
audio and raw 1000 Hz sEEG. There are 20 Mandarin word classes and six repetitions,
except one missing trial.

For a balanced primary endpoint, repetitions 1–5 form five outer folds. The incomplete
repetition 6 is development-only for adapter, audio-timing, memory, and shape smoke
checks; it is never used to fit a reported model or estimate a performance metric.
Before protocol freeze, repetitions 1–5 are numerically closed even for a pilot,
because each repetition is the held-out test set in one production outer fold. The
development runner is therefore metadata/plan-only; any engineering forward/backward
smoke uses repetition 6 and reports no classification metric.
In outer fold `k`,
repetition `k` is test, repetition `k+1` (cyclic) is validation, and the remaining
three repetitions are train.

The MEL comparator follows the VocalMind authors' released spectral definition
(16 kHz, 1024-sample/64 ms window and FFT, 320-sample/20 ms hop, 80 bins,
80–7600 Hz, STFT magnitude rather than power, and log10 amplitude clipped at
`1e-6`) but applies the same per-trial peak-absolute waveform normalization as
Whisper before the common train-only PCA-50. This matched primary target is not
called author-exact; the official no-peak extractor is a separate fidelity/sensitivity
mode. Neural-to-target regression uses every
valid unpadded 1 ms endpoint from 1.000 through 2.999 s (2000 inclusive
`[t-1000 ms,t]` windows per trial); hidden states are retained at stride 10 for a
100 Hz, 200-frame word-head trajectory. A 50 Hz variant is engineering-only and
must be labelled `fast_smoke`.

The same MEL/L3/L4/L5 upstream and hidden-sequence word head are trained per fold.
Production predeclares three MEL initializations for each outer seed (`s`,
`s+1000`, `s+2000`) and averages their softmax probabilities without seed or
subset search. Thus the primary branch-count-matched contrast is fixed L345 versus
MEL×3; single MEL and L4 are secondary comparisons.
Primary metrics are balanced accuracy, macro-F1, top-3 accuracy, and per-class recall.
The official VocalMind training code uses `test_loader` to choose its checkpoint and
does not define a separate validation split. Our port reproduces the released MEL
extractor but deliberately uses a distinct validation repetition and a held-out test
gate, so it is a stricter matched MEL-versus-Whisper comparison, not a literal
end-to-end reproduction of the authors' evaluation. Their EEG pipeline (CAR,
HGA 70–150 Hz plus low-frequency <100 Hz, 200 Hz, per-trial z-score) also differs
from our representation-independent historical 10–200 Hz/1000 Hz transform with a
train-only channel standardizer. These differences must accompany any comparison
with their published number; see `VOCALMIND_PRIMARY_RUNBOOK.md` for the executable
contract.

The fixed L345 and MEL×3 ensembles are compared using paired fold-level descriptive
statistics. Because VocalMind has one participant, folds and seeds are not presented
as independent biological replicates.

Mimed and imagined speech have no simultaneous vocal output. Zero-shot transfer of
the overt-trained neural decoders/heads to those conditions is secondary and must not
be mixed into the primary acoustic-regression result. Trialized VocalMind is not
called a true asynchronous continuous benchmark.

## 5. Test gate and immutable provenance

Every run records:

1. raw download manifest with DOI, URL, byte count, and checksum;
2. dataset index with subjects, trials, channels, rates, and events;
3. split manifest with immutable IDs and purge boundaries;
4. preprocessing/VAD manifest and hashes;
5. Git commit, complete config fingerprint, package/GPU versions, and RNG states;
6. checkpoint, reducer, prediction, and result hashes.

Test adapters may not be called before every representation and seed for that fold is
fixed. Checkpoint selection and thresholds use validation only. A crash/resume may
reuse completed training artifacts with identical fingerprints but may not silently
change a split or configuration.

## 6. Statistics and staged compute

Seeds are `1, 2, 3, 4, 42`. SWPD remains limited to development `sub-01` in v1;
VocalMind may use repetition 6 only for a non-metric engineering smoke. SWPD's
biological unit is the patient: patient-level seed means enter an exact paired
sign-flip test and patient-cluster bootstrap confidence interval. Seed variability is
reported separately and never inflates `n`. Secondary layer contrasts use Holm
correction.

VocalMind supplies a different-language repeated-word replication but only one
biological participant; inference is explicitly within-participant.

Compute proceeds in gates:

1. integrity, author-baseline reproduction, and rep6/sub-01 engineering checks;
2. freeze/tag the protocol without numerically opening VocalMind repetitions 1–5;
3. start the one immutable VocalMind run containing all five folds, all five seeds,
   and all predeclared branches; a one-epoch bounded call may measure throughput but
   cannot open test and must resume into the same run;
4. open each fold's test only after its complete 30-unit gate is satisfied;
5. aggregate exactly once from the five immutable out-of-fold prediction sets.
