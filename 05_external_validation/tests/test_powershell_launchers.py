from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_swpd_runner_avoids_fragile_inline_python_quoting() -> None:
    runner = PROJECT_ROOT / "scripts" / "run_swpd_sub01_neural_pilot.ps1"
    source = runner.read_text(encoding="utf-8")

    assert "whisper_ecog_ext.preflight" in source
    assert "& $Python -c" not in source


def test_swpd_all_runner_avoids_inline_python_and_utf8_pth_failure() -> None:
    runner = PROJECT_ROOT / "scripts" / "run_swpd_all_matched.ps1"
    source = runner.read_text(encoding="utf-8")

    assert "whisper_ecog_ext.preflight" in source
    assert "& $Python -c" not in source
    assert "Remove-Item Env:PYTHONUTF8" in source


def test_swpd_frozen_runner_avoids_inline_python_and_utf8_pth_failure() -> None:
    runner = (
        PROJECT_ROOT
        / "swpd_contextual_frozen"
        / "scripts"
        / "run_frozen.ps1"
    )
    source = runner.read_text(encoding="utf-8")

    assert "& $Python -c" not in source
    assert "Remove-Item Env:PYTHONUTF8" in source
    assert "preflight.py" in source

    start_source = runner.with_name("start_frozen_background.ps1").read_text(
        encoding="utf-8"
    )
    watch_source = runner.with_name("watch_frozen.ps1").read_text(encoding="utf-8")
    assert "-WindowStyle Hidden" in start_source
    assert "-Wait" in watch_source


def test_contextual_neural_e2e_uses_file_preflight_and_isolated_python() -> None:
    root = PROJECT_ROOT / "swpd_contextual_neural_e2e" / "scripts"
    runner = (root / "run_fit.ps1").read_text(encoding="utf-8")
    evaluator = (root / "run_evaluate_frozen.ps1").read_text(encoding="utf-8")
    launcher = (root / "start_fit_background.ps1").read_text(encoding="utf-8")
    watcher = (root / "watch_fit.ps1").read_text(encoding="utf-8")

    assert "& $Python -c" not in runner
    assert "preflight.py" in runner
    assert "'-I'" in runner
    assert "[string]$SeedCsv" in runner
    assert "[int[]]$Seeds" not in runner
    assert "preflight.py" in evaluator
    assert "-I -u" in evaluator
    assert "-WindowStyle Hidden" in launcher
    assert "'-SeedCsv', $NormalizedSeedCsv" in launcher
    assert "-Wait" in watcher
