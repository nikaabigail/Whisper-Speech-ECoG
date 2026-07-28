"""Immutable split manifests and a filesystem-backed held-out test gate."""

from __future__ import annotations

import re
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .integrity import atomic_write_json, fingerprint_json, read_json


_UNIT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,79}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class TestGateClosed(RuntimeError):
    __test__ = False

    pass


def _exact_integer(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as error:
        raise ValueError(f"{name} must be an exact integer") from error


@dataclass(frozen=True)
class TestGateAuthorization:
    """Validated proof that one exact held-out split was deliberately opened."""

    split_fingerprint: str
    protocol_fingerprint: str
    open_receipt_fingerprint: str
    held_out_sample_ids_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "split_fingerprint",
            "protocol_fingerprint",
            "open_receipt_fingerprint",
            "held_out_sample_ids_sha256",
        ):
            value = str(getattr(self, name)).lower()
            if not _SHA256.fullmatch(value):
                raise ValueError(f"{name} must be a lowercase SHA256")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class FivePairAssignment:
    """Auditable train/validation/test assignment over five adjacent trial pairs."""

    test_pair_index: int
    validation_pair_index: int
    training_pair_indices: tuple[int, int, int]


def _validate_five_adjacent_pairs(
    adjacent_trial_pairs: Sequence[Sequence[str]],
) -> tuple[tuple[str, ...], ...]:
    if len(adjacent_trial_pairs) != 5:
        raise ValueError("SWPD protocol requires exactly five adjacent trial pairs")
    pairs = tuple(
        _normalize_ids(pair, f"adjacent_trial_pair_{index}")
        for index, pair in enumerate(adjacent_trial_pairs)
    )
    flattened = tuple(sample_id for pair in pairs for sample_id in pair)
    if len(flattened) != len(set(flattened)):
        raise ValueError("sample IDs must be unique across all five adjacent trial pairs")
    pair_sizes = {len(pair) for pair in pairs}
    if len(pair_sizes) != 1:
        raise ValueError("all five adjacent trial groups must have equal size for 60/20/20")
    return pairs


def swpd_pair_assignment_for_test(test_pair_index: int) -> FivePairAssignment:
    """Assign the next pair to validation and the other three pairs to training."""

    test_index = _exact_integer(test_pair_index, "test_pair_index")
    if test_index not in range(5):
        raise ValueError("test_pair_index must be in [0, 4]")
    validation_index = (test_index + 1) % 5
    training_indices = tuple(
        index for index in range(5) if index not in (test_index, validation_index)
    )
    return FivePairAssignment(
        test_pair_index=test_index,
        validation_pair_index=validation_index,
        training_pair_indices=training_indices,
    )


def swpd_neural_pair_assignment(subject_number: int) -> FivePairAssignment:
    """Return the pre-specified full-neural assignment for one SWPD subject.

    The rule is fixed before model fitting: ``test=(subject_number-2) mod 5``;
    the following adjacent pair is validation and the remaining three are train.
    """

    subject = _exact_integer(subject_number, "subject_number")
    if subject < 1:
        raise ValueError("subject_number must be a positive integer")
    return swpd_pair_assignment_for_test((subject - 2) % 5)


def _manifest_from_pair_assignment(
    *,
    pairs: tuple[tuple[str, ...], ...],
    assignment: FivePairAssignment,
    dataset_id: str,
    protocol_id: str,
    dataset_manifest_sha256: str,
    purge_gap_seconds: float,
) -> "SplitManifest":
    train_ids = tuple(
        sample_id
        for pair_index in assignment.training_pair_indices
        for sample_id in pairs[pair_index]
    )
    return SplitManifest.create(
        dataset_id=dataset_id,
        protocol_id=protocol_id,
        split_seed=0,
        train_ids=train_ids,
        validation_ids=pairs[assignment.validation_pair_index],
        test_ids=pairs[assignment.test_pair_index],
        purge_gap_seconds=purge_gap_seconds,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )


def make_swpd_fixed_neural_split(
    *,
    subject_number: int,
    adjacent_trial_pairs: Sequence[Sequence[str]],
    dataset_id: str,
    dataset_manifest_sha256: str,
    purge_gap_seconds: float = 0.0,
) -> "SplitManifest":
    """Build the immutable 60/20/20 SWPD split used by the full neural model."""

    pairs = _validate_five_adjacent_pairs(adjacent_trial_pairs)
    assignment = swpd_neural_pair_assignment(subject_number)
    return _manifest_from_pair_assignment(
        pairs=pairs,
        assignment=assignment,
        dataset_id=dataset_id,
        protocol_id=(
            "swpd-full-neural-five-adjacent-pairs-v1/"
            f"subject-{_exact_integer(subject_number, 'subject_number'):02d}/"
            f"test-pair-{assignment.test_pair_index}"
        ),
        dataset_manifest_sha256=dataset_manifest_sha256,
        purge_gap_seconds=purge_gap_seconds,
    )


def make_swpd_rotating_linear_splits(
    *,
    adjacent_trial_pairs: Sequence[Sequence[str]],
    dataset_id: str,
    dataset_manifest_sha256: str,
    purge_gap_seconds: float = 0.0,
) -> tuple["SplitManifest", ...]:
    """Build five matched linear-analysis folds, rotating each pair through test."""

    pairs = _validate_five_adjacent_pairs(adjacent_trial_pairs)
    return tuple(
        _manifest_from_pair_assignment(
            pairs=pairs,
            assignment=swpd_pair_assignment_for_test(test_pair_index),
            dataset_id=dataset_id,
            protocol_id=(
                "swpd-matched-linear-rotating-five-pairs-v1/"
                f"test-pair-{test_pair_index}"
            ),
            dataset_manifest_sha256=dataset_manifest_sha256,
            purge_gap_seconds=purge_gap_seconds,
        )
        for test_pair_index in range(5)
    )


def _normalize_ids(values: Iterable[str], role: str) -> tuple[str, ...]:
    normalized = tuple(str(item) for item in values)
    if not normalized:
        raise ValueError(f"{role} split cannot be empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{role} split contains duplicate sample IDs")
    return normalized


@dataclass(frozen=True)
class SplitManifest:
    dataset_id: str
    protocol_id: str
    split_seed: int
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    held_out_test_ids: tuple[str, ...]
    purge_gap_seconds: float
    dataset_manifest_sha256: str
    fingerprint: str

    @classmethod
    def create(
        cls,
        *,
        dataset_id: str,
        protocol_id: str,
        split_seed: int,
        train_ids: Sequence[str],
        validation_ids: Sequence[str],
        test_ids: Sequence[str],
        purge_gap_seconds: float,
        dataset_manifest_sha256: str,
    ) -> "SplitManifest":
        train = _normalize_ids(train_ids, "train")
        validation = _normalize_ids(validation_ids, "validation")
        test = _normalize_ids(test_ids, "test")
        if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
            raise ValueError("train, validation and test sample IDs must be disjoint")
        dataset_hash = str(dataset_manifest_sha256).lower()
        if not _SHA256.fullmatch(dataset_hash):
            raise ValueError("dataset_manifest_sha256 must be a lowercase SHA256")
        if float(purge_gap_seconds) < 0:
            raise ValueError("purge_gap_seconds cannot be negative")
        payload = {
            "schema_version": 1,
            "kind": "immutable_external_validation_split",
            "dataset_id": str(dataset_id),
            "protocol_id": str(protocol_id),
            "split_seed": _exact_integer(split_seed, "split_seed"),
            "train_ids": list(train),
            "validation_ids": list(validation),
            "held_out_test_ids": list(test),
            "purge_gap_seconds": float(purge_gap_seconds),
            "dataset_manifest_sha256": dataset_hash,
        }
        return cls(
            dataset_id=payload["dataset_id"],
            protocol_id=payload["protocol_id"],
            split_seed=payload["split_seed"],
            train_ids=train,
            validation_ids=validation,
            held_out_test_ids=test,
            purge_gap_seconds=payload["purge_gap_seconds"],
            dataset_manifest_sha256=dataset_hash,
            fingerprint=fingerprint_json(payload),
        )

    def payload(self) -> dict:
        return {
            "schema_version": 1,
            "kind": "immutable_external_validation_split",
            "dataset_id": self.dataset_id,
            "protocol_id": self.protocol_id,
            "split_seed": self.split_seed,
            "train_ids": list(self.train_ids),
            "validation_ids": list(self.validation_ids),
            "held_out_test_ids": list(self.held_out_test_ids),
            "purge_gap_seconds": self.purge_gap_seconds,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
        }

    def save(self, path: Path) -> Path:
        value = self.payload()
        value["fingerprint"] = self.fingerprint
        atomic_write_json(Path(path), value, overwrite=False)
        return Path(path)

    @classmethod
    def load(cls, path: Path) -> "SplitManifest":
        value = read_json(Path(path))
        fingerprint = value.pop("fingerprint", None)
        if fingerprint != fingerprint_json(value):
            raise RuntimeError(f"Split manifest fingerprint mismatch: {path}")
        if value.get("kind") != "immutable_external_validation_split":
            raise RuntimeError(f"Unexpected split manifest kind: {path}")
        rebuilt = cls.create(
            dataset_id=value["dataset_id"],
            protocol_id=value["protocol_id"],
            split_seed=int(value["split_seed"]),
            train_ids=value["train_ids"],
            validation_ids=value["validation_ids"],
            test_ids=value["held_out_test_ids"],
            purge_gap_seconds=float(value["purge_gap_seconds"]),
            dataset_manifest_sha256=value["dataset_manifest_sha256"],
        )
        if rebuilt.fingerprint != fingerprint:
            raise RuntimeError(f"Split manifest is not canonical: {path}")
        return rebuilt


class TestGate:
    """Release test IDs only after immutable completion receipts exist."""

    __test__ = False

    def __init__(
        self,
        *,
        state_directory: Path,
        split: SplitManifest,
        required_units: Sequence[str],
        protocol_fingerprint: str,
    ) -> None:
        units = tuple(str(item) for item in required_units)
        if not units or len(units) != len(set(units)):
            raise ValueError("required_units must be a non-empty unique sequence")
        if any(not _UNIT_NAME.fullmatch(unit) for unit in units):
            raise ValueError("required unit names contain unsafe characters")
        normalized_fingerprint = str(protocol_fingerprint).lower()
        if not _SHA256.fullmatch(normalized_fingerprint):
            raise ValueError("protocol_fingerprint must be a lowercase SHA256")
        self.state_directory = Path(state_directory)
        self.split = split
        self.required_units = units
        self.protocol_fingerprint = normalized_fingerprint
        self.completions_directory = self.state_directory / "completed"
        self.open_receipt_path = self.state_directory / "test_gate_open.json"

    def training_and_validation_ids(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return self.split.train_ids, self.split.validation_ids

    def _completion_path(self, unit: str) -> Path:
        if unit not in self.required_units:
            raise ValueError(f"unit is not required by this gate: {unit}")
        return self.completions_directory / f"{unit}.json"

    def mark_completed(
        self, *, unit: str, artifact_sha256: str, run_fingerprint: str
    ) -> Path:
        artifact_hash = str(artifact_sha256).lower()
        run_hash = str(run_fingerprint).lower()
        if not _SHA256.fullmatch(artifact_hash) or not _SHA256.fullmatch(run_hash):
            raise ValueError("artifact and run fingerprints must be lowercase SHA256 values")
        path = self._completion_path(unit)
        receipt = {
            "schema_version": 1,
            "kind": "validation_fixed_training_completion",
            "unit": unit,
            "split_fingerprint": self.split.fingerprint,
            "protocol_fingerprint": self.protocol_fingerprint,
            "run_fingerprint": run_hash,
            "artifact_sha256": artifact_hash,
            "test_data_opened": False,
        }
        receipt["fingerprint"] = fingerprint_json(receipt)
        atomic_write_json(path, receipt, overwrite=False)
        return path

    def missing_units(self) -> tuple[str, ...]:
        return tuple(
            unit for unit in self.required_units if not self._completion_path(unit).is_file()
        )

    def _validated_completions(self) -> list[dict]:
        missing = self.missing_units()
        if missing:
            raise TestGateClosed(
                "held-out test gate is closed; incomplete units: " + ", ".join(missing)
            )
        receipts = []
        for unit in self.required_units:
            path = self._completion_path(unit)
            value = read_json(path)
            fingerprint = value.pop("fingerprint", None)
            if fingerprint != fingerprint_json(value):
                raise RuntimeError(f"Completion receipt fingerprint mismatch: {path}")
            if (
                value.get("schema_version") != 1
                or value.get("kind") != "validation_fixed_training_completion"
                or value.get("unit") != unit
                or value.get("split_fingerprint") != self.split.fingerprint
                or value.get("protocol_fingerprint") != self.protocol_fingerprint
                or value.get("test_data_opened") is not False
                or not _SHA256.fullmatch(str(value.get("run_fingerprint", "")))
                or not _SHA256.fullmatch(str(value.get("artifact_sha256", "")))
            ):
                raise RuntimeError(f"Completion receipt provenance mismatch: {path}")
            receipts.append(value)
        return receipts

    def _validated_open_receipt(self, completions: Sequence[dict]) -> tuple[dict, str]:
        receipt = read_json(self.open_receipt_path)
        fingerprint = receipt.pop("fingerprint", None)
        if fingerprint != fingerprint_json(receipt):
            raise RuntimeError("Test-gate receipt fingerprint mismatch")
        expected_completion_fingerprints = [
            fingerprint_json(item) for item in completions
        ]
        if (
            receipt.get("schema_version") != 1
            or receipt.get("kind") != "held_out_test_gate_open"
            or receipt.get("split_fingerprint") != self.split.fingerprint
            or receipt.get("protocol_fingerprint") != self.protocol_fingerprint
            or receipt.get("required_units") != list(self.required_units)
            or receipt.get("completion_fingerprints")
            != expected_completion_fingerprints
        ):
            raise RuntimeError("Existing test-gate receipt belongs to another experiment")
        return receipt, str(fingerprint)

    def open_test(self) -> tuple[str, ...]:
        completions = self._validated_completions()
        if self.open_receipt_path.exists():
            self._validated_open_receipt(completions)
            return self.split.held_out_test_ids

        receipt = {
            "schema_version": 1,
            "kind": "held_out_test_gate_open",
            "split_fingerprint": self.split.fingerprint,
            "protocol_fingerprint": self.protocol_fingerprint,
            "required_units": list(self.required_units),
            "completion_fingerprints": [fingerprint_json(item) for item in completions],
        }
        receipt["fingerprint"] = fingerprint_json(receipt)
        atomic_write_json(self.open_receipt_path, receipt, overwrite=False)
        return self.split.held_out_test_ids

    def test_ids(self) -> tuple[str, ...]:
        if not self.open_receipt_path.is_file():
            raise TestGateClosed("held-out test IDs are unavailable until open_test succeeds")
        return self.open_test()

    def authorization(self) -> TestGateAuthorization:
        """Return a validated evaluator token only after explicit ``open_test``."""

        if not self.open_receipt_path.is_file():
            raise TestGateClosed("held-out test gate has not been explicitly opened")
        completions = self._validated_completions()
        _, receipt_fingerprint = self._validated_open_receipt(completions)
        return TestGateAuthorization(
            split_fingerprint=self.split.fingerprint,
            protocol_fingerprint=self.protocol_fingerprint,
            open_receipt_fingerprint=receipt_fingerprint,
            held_out_sample_ids_sha256=fingerprint_json(
                list(self.split.held_out_test_ids)
            ),
        )
