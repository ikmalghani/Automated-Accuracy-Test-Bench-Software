"""Test-set metadata persistence for AATBS runs.

Each run lives in its own folder under data/:

  data/run1_20260812/
  data/run2_20260813/
  ...

Folder names are ``run{N}_{YYYYMMDD}``. The next N is max existing N + 1.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from .dataset import TestImage

RUN_FOLDER_RE = re.compile(r"^run(\d+)_(\d{8})$")
METADATA_NAME = "metadata.json"
LOG_NAME = "log.csv"
RESULTS_NAME = "analysis_results.csv"
CHART_NAME = "chart.png"  # legacy alias for the pipeline chart
CHART_PIPELINE_NAME = "chart_pipeline.png"
CHART_GATE_NAME = "chart_gate.png"
CHART_CLASSIFIER_NAME = "chart_classifier.png"

# Older runs may still use this name.
LEGACY_RESULTS_NAMES = ("results.csv",)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class RunMetadata:
    """Metadata for one accuracy test-bench run."""

    run_id: str
    created_at: str
    dataset_root: str
    images_per_class: int
    seed: Optional[int]
    total_images: int
    classes: List[str]
    images: List[TestImage] = field(default_factory=list)
    notes: str = ""
    run_number: int = 0
    folder_name: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["images"] = [img.to_dict() for img in self.images]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "RunMetadata":
        images = [TestImage.from_dict(item) for item in data.get("images", [])]
        return cls(
            run_id=str(data["run_id"]),
            created_at=str(data["created_at"]),
            dataset_root=str(data["dataset_root"]),
            images_per_class=int(data["images_per_class"]),
            seed=data.get("seed"),
            total_images=int(data.get("total_images", len(images))),
            classes=list(data.get("classes", [])),
            images=images,
            notes=str(data.get("notes", "")),
            run_number=int(data.get("run_number") or 0),
            folder_name=str(data.get("folder_name") or ""),
        )

    def mark_captured(self, index: int, captured: bool = True) -> None:
        for img in self.images:
            if img.index == index:
                img.captured = captured
                return
        raise KeyError(f"No image with index {index}")

    def captured_count(self) -> int:
        return sum(1 for img in self.images if img.captured)


def default_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


# Backwards-compatible alias used by older call sites.
def default_runs_dir() -> Path:
    return default_data_dir()


def parse_run_folder_name(name: str) -> Optional[Tuple[int, str]]:
    """Return (run_number, date_yyyymmdd) if name matches runN_YYYYMMDD."""
    match = RUN_FOLDER_RE.match(name)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def list_run_folders(data_dir: Path | None = None) -> List[Path]:
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    if not data_dir.is_dir():
        return []
    folders = []
    for path in data_dir.iterdir():
        if path.is_dir() and parse_run_folder_name(path.name):
            folders.append(path)
    return sorted(
        folders,
        key=lambda p: parse_run_folder_name(p.name)[0],  # type: ignore[index]
    )


def last_run_number(data_dir: Path | None = None) -> int:
    """Highest runN_… folder number under data/, or 0 if none."""
    highest = 0
    for path in list_run_folders(data_dir):
        parsed = parse_run_folder_name(path.name)
        if parsed and parsed[0] > highest:
            highest = parsed[0]
    return highest


def next_run_number(data_dir: Path | None = None) -> int:
    return last_run_number(data_dir) + 1


def make_run_folder_name(run_number: int, when: datetime | None = None) -> str:
    when = when or datetime.now()
    return f"run{run_number}_{when.strftime('%Y%m%d')}"


def create_run_folder(data_dir: Path | None = None) -> Path:
    """Create the next data/runN_YYYYMMDD folder and return its path."""
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    number = next_run_number(data_dir)
    folder = data_dir / make_run_folder_name(number)
    # Extremely unlikely same-day collision if deleted mid-session; bump number.
    while folder.exists():
        number += 1
        folder = data_dir / make_run_folder_name(number)
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def metadata_path(run_dir: Path) -> Path:
    return Path(run_dir) / METADATA_NAME


def log_path(run_dir: Path) -> Path:
    return Path(run_dir) / LOG_NAME


def results_path(run_dir: Path) -> Path:
    """Preferred path for writing analysis_results.csv."""
    return Path(run_dir) / RESULTS_NAME


def find_results_csv(run_dir: Path) -> Path | None:
    """Locate analysis_results.csv or a legacy results.csv if present."""
    run_dir = Path(run_dir)
    primary = run_dir / RESULTS_NAME
    if primary.is_file():
        return primary
    for legacy in LEGACY_RESULTS_NAMES:
        candidate = run_dir / legacy
        if candidate.is_file():
            return candidate
    return None


def chart_path(run_dir: Path) -> Path:
    """Legacy pipeline chart path (`chart.png`)."""
    return Path(run_dir) / CHART_NAME


def chart_paths(run_dir: Path) -> dict:
    """Separate chart files for gate, classifier, and pipeline analyses."""
    run_dir = Path(run_dir)
    return {
        "gate": run_dir / CHART_GATE_NAME,
        "classifier": run_dir / CHART_CLASSIFIER_NAME,
        "pipeline": run_dir / CHART_PIPELINE_NAME,
    }


def new_run_id(run_number: int) -> str:
    return f"run{run_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def build_run(
    dataset_root: Path,
    images: List[TestImage],
    images_per_class: int,
    seed: Optional[int] = None,
    notes: str = "",
    run_number: int = 0,
    folder_name: str = "",
) -> RunMetadata:
    classes = sorted({img.class_name for img in images})
    return RunMetadata(
        run_id=new_run_id(run_number or 0),
        created_at=_utc_now(),
        dataset_root=str(Path(dataset_root).resolve()),
        images_per_class=images_per_class,
        seed=seed,
        total_images=len(images),
        classes=classes,
        images=images,
        notes=notes,
        run_number=run_number,
        folder_name=folder_name,
    )


def save_run(meta: RunMetadata, run_dir: Path) -> Path:
    """Write metadata.json into an existing run folder. Returns that file path."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_path(run_dir)
    path.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
    return path


def create_and_save_run(
    dataset_root: Path,
    images: List[TestImage],
    images_per_class: int,
    seed: Optional[int] = None,
    notes: str = "",
    data_dir: Path | None = None,
) -> Tuple[RunMetadata, Path, Path]:
    """
    Allocate the next run folder, write metadata.json, return
    (meta, run_dir, metadata_path).
    """
    run_dir = create_run_folder(data_dir)
    parsed = parse_run_folder_name(run_dir.name)
    assert parsed is not None
    run_number, _date = parsed
    meta = build_run(
        dataset_root,
        images,
        images_per_class=images_per_class,
        seed=seed,
        notes=notes,
        run_number=run_number,
        folder_name=run_dir.name,
    )
    meta.run_id = new_run_id(run_number)
    path = save_run(meta, run_dir)
    return meta, run_dir, path


def load_run(path: Path) -> RunMetadata:
    """Load from a metadata.json file path or from a run folder path."""
    path = Path(path)
    if path.is_dir():
        path = metadata_path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = RunMetadata.from_dict(data)
    # Fill number/folder from parent dir if missing (older files).
    if not meta.folder_name or not meta.run_number:
        parent = path.parent
        parsed = parse_run_folder_name(parent.name)
        if parsed:
            meta.run_number = parsed[0]
            meta.folder_name = parent.name
    return meta


def resolve_run_dir(path: Path) -> Path:
    """Accept a run folder or a file inside it; return the run folder."""
    path = Path(path).expanduser().resolve()
    if path.is_file():
        path = path.parent
    if not path.is_dir():
        raise FileNotFoundError(f"Run folder not found: {path}")
    return path


def list_runs(data_dir: Path | None = None) -> List[Path]:
    """List run folders newest-number-last (ascending by run number)."""
    return list_run_folders(data_dir)
