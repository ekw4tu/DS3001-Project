"""Fast smoke test for the gallery + identify chain.

Bypasses InsightFace/VGG19 loading by feeding synthetic ArcFace-shaped
embeddings through build_gallery -> save/load -> identify. Runs in <1s so CI
(or you) can verify the pure-Python chain without downloading model weights.

Run:  python -m pytest tests/ -q
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ARCFACE_DIM
from src.identify import build_gallery, identify, save_gallery, load_gallery


def _fake_row(identity: str, split: str, condition: str, vec: np.ndarray, idx: int) -> dict:
    return {
        "filename": f"{identity}_{split}_{idx}.jpg",
        "identity": f"{identity}Gallery" if split == "Gallery" else f"{identity}Probe",
        "base_identity": identity,
        "condition": condition,
        "split": split,
        "model": "ArcFace",
        "embedding": vec.astype(np.float32),
    }


def _make_df(rng: np.random.Generator) -> pd.DataFrame:
    identities = ["Alice", "Bob", "Carol"]
    centers = {name: rng.normal(size=ARCFACE_DIM) for name in identities}
    rows = []
    for name, center in centers.items():
        for i in range(3):
            rows.append(_fake_row(name, "Gallery", "clean",
                                   center + 0.01 * rng.normal(size=ARCFACE_DIM), i))
        for i in range(2):
            rows.append(_fake_row(name, "Probe", "clean_probe",
                                   center + 0.01 * rng.normal(size=ARCFACE_DIM), i))
    return pd.DataFrame(rows)


def test_build_gallery_and_identify(tmp_path):
    rng = np.random.default_rng(0)
    df = _make_df(rng)

    emb, lbl, seeded = build_gallery(df, verbose=False)

    assert emb.shape == (9, ARCFACE_DIM), "3 identities * 3 Gallery rows"
    assert set(lbl) == {"Alice", "Bob", "Carol"}
    assert seeded == set(), "no auto-seed needed when every identity has Gallery rows"

    gallery_path = tmp_path / "gallery.pkl"
    save_gallery(emb, lbl, gallery_path)
    emb2, lbl2 = load_gallery(gallery_path)
    assert np.array_equal(emb, emb2)
    assert np.array_equal(lbl, lbl2)

    for target in ("Alice", "Bob", "Carol"):
        probe = df[(df["base_identity"] == target) & (df["split"] == "Probe")]
        for vec in probe["embedding"].values:
            assert identify(vec, emb2, lbl2) == target


def test_build_gallery_auto_seeds_missing_identity():
    rng = np.random.default_rng(1)
    df = _make_df(rng)
    df = df[~((df["base_identity"] == "Carol") & (df["split"] == "Gallery"))].reset_index(drop=True)

    emb, lbl, seeded = build_gallery(df, verbose=False)

    assert "Carol" in set(lbl), "missing Carol must be seeded from clean_probe"
    assert len(seeded) == 2, "both Carol probe rows used as seed"
