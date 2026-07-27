"""Read-only, lazy NWB access for the SWPD development subject.

Only ``sub-01`` is intentionally accessible in this development module.  The
confirmatory participants stay locked until the protocol and implementation
are frozen.  The adapter reads NWB's HDF5 layout directly with h5py so an
inventory does not materialize the five-minute signals in memory.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import h5py
import numpy as np


PILOT_SUBJECT = "sub-01"
LOCKED_CONFIRMATORY_SUBJECTS = tuple(f"sub-{index:02d}" for index in range(2, 11))
SUBJECT_RE = re.compile(r"^sub-\d{2}$")
DATASET_DIRECTORY = "SingleWordProductionDutch-iBIDS"


class ConfirmatoryDataLocked(PermissionError):
    """Raised before a confirmatory subject path is opened or inspected."""


class NWBLayoutError(RuntimeError):
    """The file does not expose the SWPD NWB acquisition contract."""


@dataclass(frozen=True)
class SeriesInventory:
    name: str
    shape: tuple[int, ...]
    dtype: str
    rate_hz: float
    starting_time_seconds: float
    duration_seconds: float
    unit: str | None
    timing_source: str


@dataclass(frozen=True)
class SWPDInventory:
    subject: str
    dataset_root: str
    nwb_path: str
    nwb_size_bytes: int
    nwb_version: str | None
    ieeg: SeriesInventory
    audio: SeriesInventory
    stimulus: SeriesInventory
    channels_tsv_count: int
    channel_names_unique: bool
    word_event_count: int
    fixation_event_count: int
    unique_prompt_count: int
    event_row_count: int
    event_onset_min_seconds: float | None
    event_onset_max_seconds: float | None
    nonpositive_word_duration_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualWordEvent:
    """A visual word-presentation event, explicitly not an acoustic onset."""

    trial_id: str
    trial_index: int
    onset_seconds: float
    duration_seconds: float
    prompt: str


@dataclass(frozen=True)
class SeriesSampleBounds:
    start_index: int
    stop_index: int
    actual_start_absolute_seconds: float
    actual_start_recording_relative_seconds: float


def assert_series_start_alignment(
    inventory: SWPDInventory, *, tolerance_seconds: float = 0.001
) -> dict[str, float]:
    """Verify that acquisition streams share one recording origin.

    SWPD acquisition timestamps are large absolute session-clock values, while
    ``events.tsv`` is recording-relative.  Comparing those two coordinate
    systems directly is a serious but easy-to-miss error.
    """

    if not np.isfinite(tolerance_seconds) or tolerance_seconds < 0:
        raise ValueError("tolerance_seconds must be finite and non-negative")
    reference = inventory.ieeg.starting_time_seconds
    offsets = {
        "ieeg_minus_ieeg_seconds": 0.0,
        "audio_minus_ieeg_seconds": inventory.audio.starting_time_seconds - reference,
        "stimulus_minus_ieeg_seconds": inventory.stimulus.starting_time_seconds - reference,
    }
    if any(abs(value) > tolerance_seconds for value in offsets.values()):
        raise NWBLayoutError(
            "SWPD acquisition streams do not share a recording origin within "
            f"{tolerance_seconds} s: {offsets}"
        )
    return offsets


def recording_relative_to_series_time(
    recording_relative_seconds: float, series: SeriesInventory
) -> float:
    """Map an events.tsv time to one acquisition series' absolute NWB clock."""

    relative = float(recording_relative_seconds)
    if not np.isfinite(relative) or relative < 0:
        raise ValueError("recording-relative seconds must be finite and non-negative")
    return float(series.starting_time_seconds + relative)


def recording_relative_sample_bounds(
    start_seconds: float,
    stop_seconds: float,
    series: SeriesInventory,
) -> SeriesSampleBounds:
    """Convert one relative half-open interval to contiguous series indexes."""

    tolerance = 0.5 / series.rate_hz
    if (
        float(start_seconds) > series.duration_seconds + tolerance
        or float(stop_seconds) > series.duration_seconds + tolerance
    ):
        raise ValueError(
            "recording-relative interval exceeds the acquisition duration; "
            "an absolute NWB clock may have been supplied by mistake"
        )
    absolute_start = recording_relative_to_series_time(start_seconds, series)
    absolute_stop = recording_relative_to_series_time(stop_seconds, series)
    if absolute_stop <= absolute_start:
        raise ValueError("recording-relative interval must have positive duration")
    # Nearest-sample boundaries ensure that a shared boundary has one index in
    # adjacent blocks; floor/ceil would overlap non-grid-aligned audio blocks.
    start = max(
        0,
        int(round((absolute_start - series.starting_time_seconds) * series.rate_hz)),
    )
    stop = min(
        int(series.shape[0]),
        int(round((absolute_stop - series.starting_time_seconds) * series.rate_hz)),
    )
    if stop <= start:
        raise ValueError("recording-relative interval maps to an empty sample slice")
    actual_relative = start / series.rate_hz
    return SeriesSampleBounds(
        start_index=start,
        stop_index=stop,
        actual_start_absolute_seconds=float(
            series.starting_time_seconds + actual_relative
        ),
        actual_start_recording_relative_seconds=float(actual_relative),
    )


def recording_duration_seconds(inventory: SWPDInventory) -> float:
    """Return the common usable duration in the events.tsv coordinate system."""

    assert_series_start_alignment(inventory)
    duration = min(
        inventory.ieeg.duration_seconds,
        inventory.audio.duration_seconds,
        inventory.stimulus.duration_seconds,
    )
    if not np.isfinite(duration) or duration <= 0:
        raise NWBLayoutError("SWPD recording has no positive common duration")
    return float(duration)


def assert_pilot_subject(subject: str) -> None:
    if not SUBJECT_RE.fullmatch(subject):
        raise ValueError(f"Invalid SWPD subject identifier: {subject!r}")
    if subject != PILOT_SUBJECT:
        raise ConfirmatoryDataLocked(
            f"{subject} is locked. Development code may read only {PILOT_SUBJECT}; "
            "freeze the protocol before implementing a separate confirmatory runner."
        )


def resolve_dataset_root(data_root: Path) -> Path:
    supplied = data_root.expanduser().resolve()
    candidates = (supplied, supplied / DATASET_DIRECTORY)
    for candidate in candidates:
        if (candidate / "participants.tsv").is_file():
            return candidate
    raise FileNotFoundError(
        f"Cannot find SWPD participants.tsv below {supplied}; expected {DATASET_DIRECTORY}"
    )


def subject_paths(data_root: Path, subject: str = PILOT_SUBJECT) -> dict[str, Path]:
    assert_pilot_subject(subject)
    root = resolve_dataset_root(data_root)
    ieeg_dir = root / subject / "ieeg"
    prefix = f"{subject}_task-wordProduction"
    paths = {
        "root": root,
        "nwb": ieeg_dir / f"{prefix}_ieeg.nwb",
        "channels": ieeg_dir / f"{prefix}_channels.tsv",
        "events": ieeg_dir / f"{prefix}_events.tsv",
    }
    missing = [name for name in ("nwb", "channels", "events") if not paths[name].is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {subject} files: {missing} below {ieeg_dir}")
    return paths


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _text(value.item())
    return str(value)


def _scalar(dataset: h5py.Dataset) -> float:
    value = dataset[()]
    if isinstance(value, np.ndarray):
        value = value.item()
    return float(value)


def _series_inventory(name: str, group: h5py.Group) -> SeriesInventory:
    if "data" not in group or not isinstance(group["data"], h5py.Dataset):
        raise NWBLayoutError(f"acquisition/{name}/data is missing")
    data = group["data"]
    if data.ndim < 1:
        raise NWBLayoutError(f"acquisition/{name}/data has no sample dimension")

    start = 0.0
    if "timestamps" in group:
        timestamps = group["timestamps"]
        if not isinstance(timestamps, h5py.Dataset) or len(timestamps) < 2:
            raise NWBLayoutError(f"acquisition/{name}/timestamps is too short")
        first = float(timestamps[0])
        second = float(timestamps[1])
        delta = second - first
        if not np.isfinite(delta) or delta <= 0:
            raise NWBLayoutError(f"acquisition/{name} has invalid timestamps")
        rate = 1.0 / delta
        start = first
        timing_source = "timestamps"
    elif "starting_time" in group:
        starting_time = group["starting_time"]
        if not isinstance(starting_time, h5py.Dataset):
            raise NWBLayoutError(f"acquisition/{name}/starting_time is not a dataset")
        start = _scalar(starting_time)
        rate_value = starting_time.attrs.get("rate", group.attrs.get("rate"))
        if rate_value is None:
            raise NWBLayoutError(f"acquisition/{name} has no sampling rate")
        rate = float(rate_value)
        timing_source = "starting_time.rate"
    else:
        rate_value = group.attrs.get("rate")
        if rate_value is None:
            raise NWBLayoutError(f"acquisition/{name} has neither timestamps nor rate")
        rate = float(rate_value)
        timing_source = "group.rate"
    if not np.isfinite(rate) or rate <= 0:
        raise NWBLayoutError(f"acquisition/{name} has invalid rate {rate}")

    unit_value = data.attrs.get("unit", group.attrs.get("unit"))
    unit = None if unit_value is None else _text(unit_value)
    return SeriesInventory(
        name=name,
        shape=tuple(int(value) for value in data.shape),
        dtype=str(data.dtype),
        rate_hz=rate,
        starting_time_seconds=start,
        duration_seconds=float(data.shape[0]) / rate,
        unit=unit,
        timing_source=timing_source,
    )


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_visual_word_events(data_root: Path) -> tuple[VisualWordEvent, ...]:
    """Return the 100 visual cue events for sub-01 without opening signal data."""

    paths = subject_paths(data_root, PILOT_SUBJECT)
    rows = _read_tsv(paths["events"])
    selected = [row for row in rows if row.get("trial_type") == "word"]
    events = tuple(
        VisualWordEvent(
            trial_id=f"{PILOT_SUBJECT}:trial-{index:03d}",
            trial_index=index,
            onset_seconds=float(row["onset"]),
            duration_seconds=float(row["duration"]),
            prompt=str(row.get("value", "")),
        )
        for index, row in enumerate(selected)
    )
    if len(events) != 100:
        raise NWBLayoutError(f"Expected 100 sub-01 visual word events, found {len(events)}")
    onsets = np.asarray([event.onset_seconds for event in events])
    if not np.all(np.isfinite(onsets)) or np.any(np.diff(onsets) <= 0):
        raise NWBLayoutError("Visual word-event onsets must be finite and increasing")
    if len({event.prompt for event in events}) != 100:
        raise NWBLayoutError("sub-01 must contain 100 unique visual word prompts")
    return events


class SWPDRecording:
    """Context-managed, read-only view of one SWPD pilot NWB file."""

    def __init__(self, data_root: Path, subject: str = PILOT_SUBJECT):
        assert_pilot_subject(subject)
        self.subject = subject
        self.paths = subject_paths(data_root, subject)
        self._file: h5py.File | None = None

    def __enter__(self) -> "SWPDRecording":
        if self._file is not None:
            raise RuntimeError("SWPDRecording is already open")
        self._file = h5py.File(self.paths["nwb"], mode="r", swmr=False)
        try:
            self._acquisition_group()
        except Exception:
            self._file.close()
            self._file = None
            raise
        return self

    def __exit__(self, *_: object) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            raise RuntimeError("Use SWPDRecording as a context manager")
        return self._file

    def _acquisition_group(self) -> h5py.Group:
        if "acquisition" not in self.file:
            raise NWBLayoutError("NWB file has no acquisition group")
        acquisition = self.file["acquisition"]
        if not isinstance(acquisition, h5py.Group):
            raise NWBLayoutError("NWB acquisition is not a group")
        for name in ("iEEG", "Audio", "Stimulus"):
            if name not in acquisition or not isinstance(acquisition[name], h5py.Group):
                raise NWBLayoutError(f"NWB acquisition/{name} is missing")
        return acquisition

    def _data(self, name: str) -> h5py.Dataset:
        group = self._acquisition_group()[name]
        data = group["data"]
        if not isinstance(data, h5py.Dataset):
            raise NWBLayoutError(f"acquisition/{name}/data is not a dataset")
        return data

    @staticmethod
    def _validate_slice(dataset: h5py.Dataset, start: int, stop: int) -> None:
        if not isinstance(start, int) or not isinstance(stop, int):
            raise TypeError("Sample bounds must be integers")
        if start < 0 or stop <= start or stop > dataset.shape[0]:
            raise IndexError(f"Invalid sample slice [{start}:{stop}] for {dataset.shape[0]}")

    def read_ieeg(
        self,
        start: int,
        stop: int,
        channels: slice | Sequence[int] | None = None,
    ) -> np.ndarray:
        data = self._data("iEEG")
        if data.ndim != 2:
            raise NWBLayoutError(f"iEEG must be samples x channels, got {data.shape}")
        self._validate_slice(data, start, stop)
        channel_selection: slice | Sequence[int] = slice(None) if channels is None else channels
        return np.asarray(data[start:stop, channel_selection])

    def read_audio(self, start: int, stop: int) -> np.ndarray:
        data = self._data("Audio")
        self._validate_slice(data, start, stop)
        result = np.asarray(data[start:stop])
        return result.reshape(-1)

    def read_stimulus(self, start: int, stop: int) -> np.ndarray:
        data = self._data("Stimulus")
        self._validate_slice(data, start, stop)
        return np.asarray(data[start:stop])

    def inventory(self) -> SWPDInventory:
        acquisition = self._acquisition_group()
        ieeg = _series_inventory("iEEG", acquisition["iEEG"])
        audio = _series_inventory("Audio", acquisition["Audio"])
        stimulus = _series_inventory("Stimulus", acquisition["Stimulus"])
        if len(ieeg.shape) != 2:
            raise NWBLayoutError(f"iEEG must be 2D, got {ieeg.shape}")

        channels = _read_tsv(self.paths["channels"])
        events = _read_tsv(self.paths["events"])
        channel_names = [row.get("name", "") for row in channels]
        if len(channels) != ieeg.shape[1]:
            raise NWBLayoutError(
                f"channels.tsv has {len(channels)} rows but iEEG has {ieeg.shape[1]} channels"
            )
        word_events = [row for row in events if row.get("trial_type") == "word"]
        fixation_events = [row for row in events if row.get("trial_type") == "fixation"]
        onsets = [float(row["onset"]) for row in events if row.get("onset") not in {None, ""}]
        nonpositive = sum(
            float(row.get("duration", "nan")) <= 0
            for row in word_events
            if row.get("duration") not in {None, ""}
        )
        version_value = self.file.attrs.get("nwb_version")
        version = None if version_value is None else _text(version_value)
        return SWPDInventory(
            subject=self.subject,
            dataset_root=str(self.paths["root"]),
            nwb_path=str(self.paths["nwb"]),
            nwb_size_bytes=self.paths["nwb"].stat().st_size,
            nwb_version=version,
            ieeg=ieeg,
            audio=audio,
            stimulus=stimulus,
            channels_tsv_count=len(channels),
            channel_names_unique=len(channel_names) == len(set(channel_names)),
            word_event_count=len(word_events),
            fixation_event_count=len(fixation_events),
            unique_prompt_count=len({row.get("value", "") for row in word_events}),
            event_row_count=len(events),
            event_onset_min_seconds=min(onsets) if onsets else None,
            event_onset_max_seconds=max(onsets) if onsets else None,
            nonpositive_word_duration_count=nonpositive,
        )


def inventory_pilot(data_root: Path) -> SWPDInventory:
    paths = subject_paths(data_root, PILOT_SUBJECT)
    before = {
        name: (path.stat().st_size, path.stat().st_mtime_ns)
        for name, path in paths.items()
        if name != "root"
    }
    with SWPDRecording(data_root, PILOT_SUBJECT) as recording:
        result = recording.inventory()
    after = {
        name: (path.stat().st_size, path.stat().st_mtime_ns)
        for name, path in paths.items()
        if name != "root"
    }
    if before != after:
        raise RuntimeError("A raw SWPD file changed during read-only inventory")
    return result
