"""PlantVillage dataset scanning and test-set sampling."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Disease labels used by the ACLIS / STM32 disease model.
DISEASE_CLASSES = ("bacterial", "fungal", "healthy", "pest", "viral")


@dataclass
class TestImage:
    """One image entry in a generated test set."""

    index: int
    path: str
    filename: str
    class_name: str
    captured: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TestImage":
        return cls(
            index=int(data["index"]),
            path=str(data["path"]),
            filename=str(data["filename"]),
            class_name=str(data["class_name"]),
            captured=bool(data.get("captured", False)),
        )


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def is_readable_image(path: Path) -> bool:
    """True if Pillow can identify and verify the file as an image."""
    if not _is_image(path):
        return False
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except OSError:
        return False


def discover_classes(dataset_root: Path) -> Dict[str, List[Path]]:
    """
    Discover class folders under a PlantVillage-style root.

    Supports:
      root/class_name/*.jpg
      root/train/class_name/*.jpg  (or val/test)
    """
    dataset_root = Path(dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")

    classes: Dict[str, List[Path]] = {}

    # Prefer explicit split folders if present.
    split_dirs = [
        p
        for p in dataset_root.iterdir()
        if p.is_dir() and p.name.lower() in {"train", "val", "valid", "validation", "test"}
    ]

    search_roots = split_dirs if split_dirs else [dataset_root]

    for root in search_roots:
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            images = sorted(p for p in class_dir.iterdir() if _is_image(p))
            if not images:
                continue
            name = class_dir.name
            classes.setdefault(name, []).extend(images)

    if not classes:
        raise ValueError(
            "No class folders with images found. Expected PlantVillage layout:\n"
            "  <dataset>/<class_name>/*.jpg\n"
            "or\n"
            "  <dataset>/train/<class_name>/*.jpg"
        )

    return classes


def normalize_label(class_name: str) -> str:
    """
    Map a PlantVillage folder name toward the 5 disease labels when possible.
    Falls back to the folder name (lowercased) otherwise.
    """
    lowered = class_name.lower().replace(" ", "_").replace("-", "_")

    # Exact match first.
    if lowered in DISEASE_CLASSES:
        return lowered

    # Common PlantVillage cues (folder often looks like Crop___Disease).
    if "healthy" in lowered:
        return "healthy"
    if "bacter" in lowered:
        return "bacterial"
    if "virus" in lowered or "viral" in lowered or "mosaic" in lowered:
        return "viral"
    if any(
        tip in lowered
        for tip in (
            "blight",
            "rust",
            "scab",
            "mildew",
            "spot",
            "mold",
            "rot",
            "anthracnose",
            "canker",
            "leaf_mold",
            "septoria",
            "alternaria",
            "cercospora",
            "fung",
        )
    ):
        return "fungal"
    if "pest" in lowered or "mite" in lowered or "spider" in lowered:
        return "pest"

    return lowered


def sample_test_set(
    dataset_root: Path,
    images_per_class: int,
    seed: int | None = None,
    class_filter: Sequence[str] | None = None,
    use_normalized_labels: bool = True,
) -> List[TestImage]:
    """
    Randomly sample up to `images_per_class` images from each class folder.

    When `use_normalized_labels` is True, samples are grouped by normalized
    disease label (bacterial/fungal/healthy/pest/viral) so multiple PlantVillage
    folders that map to the same disease share one quota.
    """
    if images_per_class < 1:
        raise ValueError("images_per_class must be >= 1")

    rng = random.Random(seed)
    raw = discover_classes(dataset_root)

    # Group source images by output label.
    grouped: Dict[str, List[Path]] = {}
    for folder_name, paths in raw.items():
        label = normalize_label(folder_name) if use_normalized_labels else folder_name
        if class_filter and label not in class_filter and folder_name not in class_filter:
            continue
        grouped.setdefault(label, []).extend(paths)

    if not grouped:
        raise ValueError("No classes left after filtering.")

    selected: List[TestImage] = []
    for label in sorted(grouped.keys()):
        pool = list(grouped[label])
        rng.shuffle(pool)
        take: List[Path] = []
        for path in pool:
            if len(take) >= images_per_class:
                break
            if is_readable_image(path):
                take.append(path)
        for path in take:
            selected.append(
                TestImage(
                    index=0,  # filled after shuffle
                    path=str(path.resolve()),
                    filename=path.name,
                    class_name=label,
                )
            )

    # Present images in a mixed order so capture is not class-blocked.
    rng.shuffle(selected)
    for i, item in enumerate(selected, start=1):
        item.index = i

    return selected
