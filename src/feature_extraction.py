"""ArcFace and VGG19 feature extraction.

Lifted from Stage 2 cells 7 (ArcFace) and 10 (VGG19). Unified into functions
that accept a list of image paths so the same code serves full-dataset
extraction (build_gallery), incremental additions (add_identity), and
ad-hoc single-image calls.
"""
from pathlib import Path
from typing import Iterable
import os

import cv2
import numpy as np
from tqdm import tqdm

from .config import IMG_EXTENSIONS
from .metadata import parse_base_identity, parse_condition


def walk_image_tasks(base_path: Path, splits=("Gallery", "Probe")) -> list[tuple]:
    """Return (split, folder_path, filename) tuples for all images under base_path.

    Accepts 'Gallery' or 'Gallery ' (trailing space) to match the original
    dataset layout.
    """
    tasks = []
    for split in splits:
        for candidate in (base_path / split, base_path / f"{split} "):
            if candidate.exists():
                for root, _, files in os.walk(candidate):
                    for name in files:
                        if name.lower().endswith(IMG_EXTENSIONS):
                            tasks.append((split, root, name))
                break
    return tasks


def init_arcface(use_gpu: bool = False):
    """Load buffalo_l (detection + alignment + ArcFace embedding).

    WHY buffalo_l: matches the Stage 2 choice so gallery vectors stay
    compatible with previously saved embeddings.
    """
    from insightface.app import FaceAnalysis

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    )
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))
    return app


def init_vgg19():
    """Load VGG19 pretrained on ImageNet, expose the 4096-d fc2 output."""
    from tensorflow.keras.applications.vgg19 import VGG19
    from tensorflow.keras.models import Model

    base = VGG19(weights="imagenet", include_top=True)
    return Model(inputs=base.input, outputs=base.get_layer("fc2").output)


def extract_arcface(tasks: Iterable[tuple], app) -> list[dict]:
    """Return [{filename, identity, base_identity, condition, split, embedding}, ...]."""
    records = []
    for split, root, name in tqdm(list(tasks), desc="ArcFace"):
        img = cv2.imread(os.path.join(root, name))
        if img is None:
            continue
        faces = app.get(img)
        if not faces:
            continue
        folder = os.path.basename(root)
        records.append({
            "filename": name,
            "identity": folder,
            "base_identity": parse_base_identity(folder, name),
            "condition": parse_condition(folder, name, split),
            "split": split.strip(),
            "model": "ArcFace",
            "embedding": faces[0].embedding.astype(np.float32),
        })
    return records


def extract_vgg19(tasks: Iterable[tuple], app, vgg_model) -> list[dict]:
    """VGG19 features from face crops detected by InsightFace.

    WHY use InsightFace for cropping: VGG19 has no face detector of its own;
    reusing the ArcFace detector keeps the crop consistent between the two
    feature sets so comparisons are apples-to-apples.
    """
    from tensorflow.keras.applications.vgg19 import preprocess_input

    records = []
    for split, root, name in tqdm(list(tasks), desc="VGG19"):
        img = cv2.imread(os.path.join(root, name))
        if img is None:
            continue
        faces = app.get(img)
        if not faces:
            continue
        x1, y1, x2, y2 = faces[0].bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        batch = preprocess_input(np.expand_dims(resized, axis=0))
        feat = vgg_model.predict(batch, verbose=0)[0].astype(np.float32)

        folder = os.path.basename(root)
        records.append({
            "filename": name,
            "identity": folder,
            "base_identity": parse_base_identity(folder, name),
            "condition": parse_condition(folder, name, split),
            "split": split.strip(),
            "model": "VGG19",
            "embedding": feat,
        })
    return records


def extract_arcface_for_folder(folder: Path, app, label: str, condition: str = "clean",
                                split: str = "Gallery") -> list[dict]:
    """Extract ArcFace vectors for every image in one folder under a fixed label.

    Used by add_identity.py so the caller controls the canonical name rather
    than relying on filename heuristics.
    """
    records = []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(IMG_EXTENSIONS):
            continue
        img = cv2.imread(str(folder / name))
        if img is None:
            continue
        faces = app.get(img)
        if not faces:
            continue
        records.append({
            "filename": name,
            "identity": folder.name,
            "base_identity": label,
            "condition": condition,
            "split": split,
            "model": "ArcFace",
            "embedding": faces[0].embedding.astype(np.float32),
        })
    return records
