from pathlib import Path

from whisper_ecog_ext.source_identity import capture_source_identity


def test_source_identity_changes_when_executable_file_changes(tmp_path: Path) -> None:
    (tmp_path / "src" / "package").mkdir(parents=True)
    source = tmp_path / "src" / "package" / "runner.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    first = capture_source_identity(tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second = capture_source_identity(tmp_path)
    assert first["files_fingerprint"] != second["files_fingerprint"]
    assert first["fingerprint"] != second["fingerprint"]
