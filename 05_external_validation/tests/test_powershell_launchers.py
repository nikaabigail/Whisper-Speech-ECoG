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
