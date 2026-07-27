# SWPD development gate (`sub-01` only)

This module prepares the public **Dataset of Speech Production in intracranial
Electroencephalography** (SWPD, OSF `nrgx6`) without changing any historical
Ivanova/Procenko code.  It deliberately refuses to open `sub-02` through
`sub-10` until the external-validation protocol is frozen.

Primary sources:

- article: <https://doi.org/10.1038/s41597-022-01542-9>
- dataset: <https://doi.org/10.17605/OSF.IO/NRGX6>
- author code: <https://github.com/neuralinterfacinglab/SingleWordProductionDutch>

## 1. Verify the implementation without participant data

From `05_external_validation`:

```powershell
py -3.10 -m unittest discover -s tests -p "test_swpd.py" -v
```

The tests create a small synthetic NWB/HDF5 fixture, verify lazy read-only
inventory, exercise the author high-gamma/MEL path, confirm that `sub-02` is
rejected before path lookup, and check ZIP traversal protection and top-level
layout.

## 2. Download and verify SWPD

Use a data directory outside the Git checkout.  The command resumes a partial
download, verifies the pinned 2,794,936,886-byte ZIP by SHA-256, safely extracts
it, and writes a receipt alongside the extracted dataset.

```powershell
py -3.10 .\swpd_download.py --destination "C:\WhisperECoG\swpd"
```

Expected SHA-256:

```text
015bc9c565c3dbdc7259c01be54f62b3346cbd7dc5cec8156eb718f64b6cbcd9
```

OSF's HEAD chain currently reports an inconsistent smaller `Content-Length`.
The discrepancy is recorded in `manifests/swpd_osf_nrgx6.json`; the downloader
does not trust HEAD for integrity.

## 3. Run the read-only pilot inventory

```powershell
py -3.10 .\swpd_inventory.py `
  --data-root "C:\WhisperECoG\swpd\extracted\SingleWordProductionDutch-iBIDS" `
  --output "C:\WhisperECoG\swpd\derived\sub01_inventory.json"
```

No signal array is loaded in full by the inventory.  The adapter opens the NWB
file in HDF5 read-only mode and checks that raw file sizes and modification
times are unchanged after inspection.

## 4. Reproduce the authors' MEL technical validation

Install the locked environment first, then run:

```powershell
py -3.10 .\swpd_author_mel.py `
  --data-root "C:\WhisperECoG\swpd\extracted\SingleWordProductionDutch-iBIDS" `
  --output-dir "C:\WhisperECoG\swpd\runs\author_mel_sub01" `
  --subject sub-01 `
  --seed 0 `
  --randomizations 1000
```

The executable protocol is the authors' 70--170 Hz envelope, 50 ms/10 ms
frames, nine contexts from -200 to +200 ms, fold-train standardization and
PCA-50, OLS to 23-bin log-MEL, sequential 10-fold CV, and circular-shift null.
The released NWB timestamp clock is approximately 47,999.19 Hz for `sub-01`,
but the authors' code explicitly processes it as 48,000 Hz before factor-3
decimation. Both the measured and protocol-assumed rates are recorded.
The JSON result carries `exact_reproduction: true` and records every numerical
compatibility modernization.  Griffin--Lim waveform synthesis is omitted
because it does not alter the reconstruction metric.

This output is a **development fidelity check**, not a confirmatory result.
Do not add an unlock switch or run another subject before protocol freeze.

## 5. Development matched-linear comparison

This is separate from the exact 23-bin author reproduction. It compares MEL80
and pinned Whisper-base L3/L4/L5 with one controlled protocol:

- block-local high-gamma extraction on a common 20 ms grid;
- five adjacent visual blocks, each containing 20 trials;
- visual events define block boundaries only and are not speech onsets;
- for each rotating fold: three blocks train, the following block validates,
  and one block tests;
- one fold-train neural StandardScaler/PCA50 is reused by every target;
- each MEL80/Whisper target receives its own fold-train PCA50 whitening;
- identical ordinary least-squares regression for every target;
- standardized MSE, explained variance, and Fisher-z component correlation.

Whisper L3/L4/L5 are collected together in one encoder forward per 30-second
chunk. Completed block caches are checksummed and reused after interruption.

```powershell
py -3.10 .\swpd_matched_linear.py `
  --data-root "C:\WhisperECoG\SWPD\extracted\SingleWordProductionDutch-iBIDS" `
  --cache-dir "C:\WhisperECoG\SWPD\derived\matched_linear_sub01" `
  --output-dir "C:\WhisperECoG\SWPD\runs\matched_linear_sub01_v1" `
  --subject sub-01 `
  --device cuda
```

Speech-only metrics remain `null` unless an independently audited audio TSV is
supplied. The accepted format is tab-separated
`onset_seconds`, `offset_seconds`, `label_source`; `label_source` must be
`audio_manual` or `audio_vad_audited`. Cue intervals are deliberately rejected
as speech labels.

This matched-linear path is offline/noncausal: high-gamma uses zero-phase
filtering within each block and Whisper targets are bidirectional within their
30-second chunk. It is a representation comparison, not an online asynchronous
claim.

## 6. Full-neural `sub-01` regression pilot

The production pilot trains the common `OneSecondEcogEncoder` from scratch on
the same neural input for every acoustic representation. Its fixed path is:

1. keep all 127 `sub-01` SEEG rows in `channels.tsv` order (there are no
   explicit bad flags);
2. independently inside each of five visual blocks, apply zero-phase notches at
   50/100/150 Hz and a 10--200 Hz Butterworth band-pass;
3. deterministic polyphase resampling from 1024 to exactly 1000 Hz; CAR is
   explicitly disabled, matching the historical/SWPD baseline;
4. fit one channel standardizer on training blocks only, then transform each
   whole block once before lazy overlapping-window slicing;
5. for every 1 ms target time, slice the inclusive lag-geometry window
   `[t-1000 ms, t]` with exactly 1001 samples and no padding;
6. fit a separate train-only `StandardScaler -> PCA50(whiten=True)` target for
   MEL80, Whisper L3, L4, and L5;
7. select checkpoints only by validation MSE, then open the regression-only
   held-out test gate after every required model is fixed.

The regression grid is 1000 Hz. A later hidden-sequence head uses fixed stride
10, yielding a 100 Hz hidden trajectory. `-FastSmoke` intentionally uses only
50 Hz and at most two epochs; its output is labelled diagnostic and must not be
reported. Although input windows never use samples after `t`, the fixed
zero-phase preprocessing is offline/noncausal and is disclosed as such.

The default branch-count-matched run trains three predeclared MEL initializations
(seeds 4, 1004, 2004) plus one model each for L3/L4/L5 (seed 4). It never picks
the best MEL replica. A single MEL model is available only through the explicit
development flag.

From `05_external_validation`:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_swpd_sub01_neural_pilot.ps1 `
  -DataRoot "C:\WhisperECoG\SWPD\extracted" `
  -CacheDir "C:\WhisperECoG_Work\SWPD\cache_1000hz" `
  -RunDir "C:\WhisperECoG_Work\SWPD\runs\seed4_v1"
```

The script performs a CUDA check, creates deterministic audio-only VAD
candidates, builds checksummed block caches, and resumes each model from the
last completed epoch. Stopping the foreground process during an epoch loses at
most that unfinished epoch; rerunning the identical command resumes from the
atomic checkpoint. Caches, checkpoints, and raw data remain outside Git.

For a short engineering check, use separate directories so a 50 Hz fingerprint
cannot be confused with production:

```powershell
.\scripts\run_swpd_sub01_neural_pilot.ps1 `
  -FastSmoke `
  -CacheDir "C:\WhisperECoG_Work\SWPD\cache_smoke_50hz" `
  -RunDir "C:\WhisperECoG_Work\SWPD\runs\smoke_50hz"
```

The audio VAD TSV is deliberately labelled
`audio_energy_candidate_unreviewed`. Visual cues are not read by that command.
Regression may finish before manual annotation, but event metrics, continuous
heads, and the fixed L3+L4+L5 probability ensemble remain code-gated until an
auditor supplies both an audio-derived interval TSV and a bound approval
receipt.

## 7. Timebase correction caught before neural training

SWPD NWB streams use a large absolute session clock (`~9062565.623 s` for
`sub-01`), while `events.tsv` uses recording-relative seconds (`0...300`). An
early matched-block implementation mixed the two. It was corrected before a
real matched/neural run by adding an explicit relative-event to absolute-series
conversion, checking stream-start agreement, and keeping cached/evaluated frame
times recording-relative. Adjacent block bounds now use one nearest-sample,
half-open boundary, so neither iEEG nor audio blocks overlap.

The real metadata/bounds smoke for `sub-01` passed: all five iEEG/audio block
boundaries were contiguous and inside their arrays; the audio-to-iEEG stream
start difference was `3.45 microseconds`; every first 1001-sample window started
at index zero and every last window stayed at least one second inside its block.
No Whisper extraction or training was run by that smoke.

Run all SWPD synthetic and CPU integration tests with:

```powershell
py -3.10 -m unittest discover -s tests -p "test_swpd*.py" -v
```
