"""Train-only StandardScaler/PCA artifacts without executable pickle payloads."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .integrity import fingerprint_json, read_json, sha256_file


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sample_id_hash(sample_ids: Sequence[str]) -> str:
    normalized = [str(item) for item in sample_ids]
    if len(normalized) != len(set(normalized)):
        raise ValueError("training sample IDs must be unique")
    return fingerprint_json(normalized)


@dataclass(frozen=True)
class ReducerArtifact:
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    pca_mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    whiten: bool
    seed: int
    train_sample_count: int
    train_sample_ids_sha256: str

    @property
    def input_dim(self) -> int:
        return int(self.scaler_mean.shape[0])

    @property
    def output_dim(self) -> int:
        return int(self.components.shape[0])

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(
                f"features must have shape (n, {self.input_dim}); got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("features contain NaN or Infinity")
        standardized = (values - self.scaler_mean) / self.scaler_scale
        reduced = (standardized - self.pca_mean) @ self.components.T
        if self.whiten:
            epsilon = np.finfo(self.explained_variance.dtype).eps
            reduced = reduced / np.sqrt(np.maximum(self.explained_variance, epsilon))
        return reduced.astype(np.float32)

    def manifest_payload(self) -> dict:
        return {
            "schema_version": 1,
            "kind": "train_only_standard_scaler_pca",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "whiten": bool(self.whiten),
            "seed": int(self.seed),
            "train_sample_count": int(self.train_sample_count),
            "train_sample_ids_sha256": self.train_sample_ids_sha256,
            "sklearn_version": sklearn.__version__,
        }

    def save(self, directory: Path) -> Path:
        """Create an immutable directory containing JSON metadata and numeric NPZ."""

        directory = Path(directory)
        if directory.exists():
            raise FileExistsError(f"Reducer artifact already exists: {directory}")
        directory.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{directory.name}.partial-", dir=directory.parent)
        )
        try:
            arrays_path = temporary / "reducer_arrays.npz"
            np.savez_compressed(
                arrays_path,
                scaler_mean=np.asarray(self.scaler_mean, dtype=np.float64),
                scaler_scale=np.asarray(self.scaler_scale, dtype=np.float64),
                pca_mean=np.asarray(self.pca_mean, dtype=np.float64),
                components=np.asarray(self.components, dtype=np.float64),
                explained_variance=np.asarray(self.explained_variance, dtype=np.float64),
            )
            payload = self.manifest_payload()
            payload["arrays_file"] = arrays_path.name
            payload["arrays_sha256"] = sha256_file(arrays_path)
            payload["fingerprint"] = fingerprint_json(payload)
            (temporary / "manifest.json").write_text(
                __import__("json").dumps(
                    payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            if directory.exists():
                raise FileExistsError(f"Reducer artifact already exists: {directory}")
            os.rename(temporary, directory)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return directory

    @classmethod
    def load(cls, directory: Path) -> "ReducerArtifact":
        directory = Path(directory)
        payload = read_json(directory / "manifest.json")
        expected_fingerprint = payload.pop("fingerprint", None)
        if expected_fingerprint != fingerprint_json(payload):
            raise RuntimeError(f"Reducer manifest fingerprint mismatch: {directory}")
        if payload.get("kind") != "train_only_standard_scaler_pca":
            raise RuntimeError(f"Unexpected reducer artifact kind: {directory}")
        arrays_path = directory / str(payload["arrays_file"])
        if sha256_file(arrays_path) != payload.get("arrays_sha256"):
            raise RuntimeError(f"Reducer numeric payload checksum mismatch: {arrays_path}")
        sample_hash = str(payload.get("train_sample_ids_sha256", ""))
        if not _SHA256.fullmatch(sample_hash):
            raise RuntimeError("Reducer train-sample hash is invalid")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            artifact = cls(
                scaler_mean=np.asarray(arrays["scaler_mean"], dtype=np.float64),
                scaler_scale=np.asarray(arrays["scaler_scale"], dtype=np.float64),
                pca_mean=np.asarray(arrays["pca_mean"], dtype=np.float64),
                components=np.asarray(arrays["components"], dtype=np.float64),
                explained_variance=np.asarray(
                    arrays["explained_variance"], dtype=np.float64
                ),
                whiten=bool(payload["whiten"]),
                seed=int(payload["seed"]),
                train_sample_count=int(payload["train_sample_count"]),
                train_sample_ids_sha256=sample_hash,
            )
        if artifact.input_dim != int(payload["input_dim"]):
            raise RuntimeError("Reducer input dimension disagrees with manifest")
        if artifact.output_dim != int(payload["output_dim"]):
            raise RuntimeError("Reducer output dimension disagrees with manifest")
        return artifact


def fit_train_only_reducer(
    features: np.ndarray,
    sample_ids: Sequence[str],
    *,
    n_components: int = 50,
    whiten: bool = True,
    seed: int = 42,
    split_role: str,
) -> ReducerArtifact:
    """Fit on an explicitly declared training split and return a portable artifact."""

    if split_role != "train":
        raise ValueError("Reducer fitting is allowed only with split_role='train'")
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("features must be a non-empty two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("training features contain NaN or Infinity")
    if len(sample_ids) != values.shape[0]:
        raise ValueError("one unique sample ID is required for every training row")
    sample_hash = _sample_id_hash(sample_ids)
    maximum = min(values.shape)
    if not 1 <= int(n_components) <= maximum:
        raise ValueError(f"n_components must be in [1, {maximum}]")

    scaler = StandardScaler(copy=True).fit(values)
    standardized = scaler.transform(values)
    pca = PCA(
        n_components=int(n_components),
        whiten=bool(whiten),
        svd_solver="full",
        random_state=int(seed),
    ).fit(standardized)
    return ReducerArtifact(
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
        pca_mean=np.asarray(pca.mean_, dtype=np.float64),
        components=np.asarray(pca.components_, dtype=np.float64),
        explained_variance=np.asarray(pca.explained_variance_, dtype=np.float64),
        whiten=bool(whiten),
        seed=int(seed),
        train_sample_count=int(values.shape[0]),
        train_sample_ids_sha256=sample_hash,
    )
