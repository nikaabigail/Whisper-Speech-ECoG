# External validation on two public invasive-speech datasets

This directory is a new, self-contained validation track. It does not change the
historical Ivanova/Procenko pipelines or reuse their patient-specific weights.

The two-computer design is:

| Host | Dataset | Primary role |
|---|---|---|
| RTX 5070 laptop | SingleWordProductionDutch (SWPD) | 10-patient neural-to-acoustic reconstruction and true continuous speech-event detection |
| RTX 3060 Ti-class desktop | VocalMind | repeated 20-class Mandarin word decoding and fixed L3+L4+L5 ensemble |

Both hosts use the same code commit, frozen `openai/whisper-base`, layers L3/L4/L5,
one-second neural context, train-only transforms, test gate, and equal-weight
probability ensemble. Dataset-specific adapters and downstream heads are different
because the datasets support different scientific questions.

## Scientific guardrails

- SWPD has 100 unique words per subject and therefore is **not** used for ordinary
  within-subject 100-class word classification.
- Visual cue timestamps are not acoustic speech onsets. Continuous labels must be
  derived independently from audio and manually audited.
- VocalMind is trialized, so it is not presented as a free-running asynchronous test.
- L3+L4+L5 is fixed before external test access; no subset search is allowed.
- Test thresholds and the `Recall ~= 0.40` operating point are selected on validation.
- Patient, not optimizer seed or frame, is the biological statistical unit.

The frozen v1 analysis contract is in [PROTOCOL_DRAFT.md](PROTOCOL_DRAFT.md). The
filename is retained for stable links; its contents and the production config are
bound to the exact clean Git commit used for a run.
The two-machine installation and launch sequence is written in Russian in
[RUNBOOK_RU.md](RUNBOOK_RU.md). Real-data checks, decisions, and bugs caught before
launch are recorded chronologically in [LAB_JOURNAL.md](LAB_JOURNAL.md).

## Clean Windows 11 setup

After cloning the repository, open PowerShell in this directory.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap_windows.ps1 -Dataset vocalmind -DataRoot "D:\WhisperECoG\data" -InstallSystemTools
```

For the current SWPD host:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap_windows.ps1 -Dataset swpd -DataRoot "C:\WhisperECoG\data"
```

The bootstrap installs Python 3.10/Git only when requested, creates an isolated
virtual environment, installs the CUDA 12.8 PyTorch wheel, and executes a real GPU
forward/backward check. A standalone CUDA Toolkit is not required.

The required order is:

1. download plus checksum verification;
2. read-only dataset inventory;
3. author-MEL fidelity check;
4. rep6-only non-metric GPU smoke;
5. checkout the published protocol-freeze commit with a clean worktree;
6. run all five VocalMind folds and all five seeds inside one immutable output root.

SWPD `sub-02`–`sub-10` are not opened by this release. They require a separate
confirmatory runner after the `sub-01` development and speech-boundary audits.
