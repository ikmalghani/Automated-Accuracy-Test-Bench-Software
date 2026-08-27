"""Image dataset scanning and test-set sampling."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Hold-out splits used for test-set sampling. Train is never sampled.
HOLD_OUT_SPLIT_NAMES = frozenset({"val", "valid", "validation", "test"})
TRAIN_SPLIT_NAMES = frozenset({"train"})
SPLIT_FOLDER_NAMES = HOLD_OUT_SPLIT_NAMES | TRAIN_SPLIT_NAMES


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


def _split_dirs(dataset_root: Path, names: Iterable[str]) -> List[Path]:
    wanted = {n.lower() for n in names}
    return sorted(
        p
        for p in dataset_root.iterdir()
        if p.is_dir() and p.name.lower() in wanted
    )


def hold_out_split_dirs(dataset_root: Path) -> List[Path]:
    """Validation and test split folders under a dataset root, if any."""
    return _split_dirs(dataset_root, HOLD_OUT_SPLIT_NAMES)


def _file_identity(path: Path) -> Tuple:
    """Identity key so hardlinks / duplicate paths count as one image."""
    try:
        resolved = path.resolve()
        st = resolved.stat()
        return (st.st_dev, st.st_ino)
    except OSError:
        return ("path", str(path))


def _dedupe_paths(paths: Sequence[Path]) -> List[Path]:
    seen: Set[Tuple] = set()
    unique: List[Path] = []
    for path in paths:
        key = _file_identity(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def discover_classes(dataset_root: Path) -> Dict[str, List[Path]]:
    """
    Discover class folders under a dataset root.

    Each subfolder name is a class. Supports:
      root/class_name/*.jpg
      root/val|validation|test/class_name/*.jpg

    When train/val/test split folders exist, only validation and test are
    used. Train is never included. Duplicate files (same path or inode)
    are collapsed so each image appears once per class.
    """
    dataset_root = Path(dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")

    all_splits = _split_dirs(dataset_root, SPLIT_FOLDER_NAMES)
    hold_out = hold_out_split_dirs(dataset_root)

    if all_splits and not hold_out:
        raise ValueError(
            "Dataset has a train split but no validation or test folder.\n"
            "Test-set images are taken only from val/validation/test, never train.\n"
            "Expected:\n"
            "  <dataset>/val/<class_name>/*.jpg\n"
            "  <dataset>/test/<class_name>/*.jpg"
        )

    search_roots = hold_out if hold_out else [dataset_root]

    classes: Dict[str, List[Path]] = {}
    for root in search_roots:
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            images = sorted(p for p in class_dir.iterdir() if _is_image(p))
            if not images:
                continue
            name = class_dir.name
            classes.setdefault(name, []).extend(images)

    for name, paths in list(classes.items()):
        classes[name] = _dedupe_paths(paths)

    if not classes:
        raise ValueError(
            "No class folders with images found. Expected:\n"
            "  <dataset>/<class_name>/*.jpg\n"
            "or\n"
            "  <dataset>/val/<class_name>/*.jpg  and/or  "
            "<dataset>/test/<class_name>/*.jpg\n"
            "(train folders are not used for test-set sampling)"
        )

    return classes


def clone_test_images(source: Sequence[TestImage]) -> List[TestImage]:
    """Copy a previous run's test-set entries, resetting capture flags."""
    cloned: List[TestImage] = []
    for img in source:
        cloned.append(
            TestImage(
                index=int(img.index),
                path=str(img.path),
                filename=str(img.filename),
                class_name=str(img.class_name),
                captured=False,
            )
        )
    cloned.sort(key=lambda item: item.index)
    return cloned


def sample_test_set(
    dataset_root: Path,
    images_per_class: int,
    seed: int | None = None,
    class_filter: Sequence[str] | None = None,
) -> List[TestImage]:
    """
    Randomly sample exactly `images_per_class` unique images from each class.

    Images come only from validation and test folders when those splits exist.
    Train is never sampled. Each selected file is unique within the test set
    (no repeated path / inode, including across val and test).
    """
    if images_per_class < 1:
        raise ValueError("images_per_class must be >= 1")

    rng = random.Random(seed)
    raw = discover_classes(dataset_root)

    grouped: Dict[str, List[Path]] = {}
    for folder_name, paths in raw.items():
        if class_filter and folder_name not in class_filter:
            continue
        grouped.setdefault(folder_name, []).extend(paths)

    if not grouped:
        raise ValueError("No classes left after filtering.")

    selected: List[TestImage] = []
    used_identities: Set[Tuple] = set()
    for label in sorted(grouped.keys()):
        pool = _dedupe_paths(grouped[label])
        rng.shuffle(pool)
        take: List[Path] = []
        for path in pool:
            if len(take) >= images_per_class:
                break
            identity = _file_identity(path)
            if identity in used_identities:
                continue
            if is_readable_image(path):
                take.append(path)
                used_identities.add(identity)
        if len(take) < images_per_class:
            raise ValueError(
                f"Class '{label}' has only {len(take)} unique readable images "
                f"in validation/test (requested {images_per_class}). "
                "Lower images-per-class or add more val/test images."
            )
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
