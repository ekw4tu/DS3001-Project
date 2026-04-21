"""Build the ArcFace gallery and identify unknown vectors.

build_gallery auto-seeds any identity that has no Gallery/ subfolder from its
clean_probe rows, so identities present only under Probe/ are not silently
dropped from the gallery.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .config import GALLERY_PKL


def build_gallery(embeddings_df: pd.DataFrame, verbose: bool = True) -> tuple[np.ndarray, np.ndarray, set[str]]:
    """Return (embeddings, labels, seeded_filenames).

    seeded_filenames is returned so callers can exclude those rows from any
    held-out evaluation to avoid leakage.
    """
    arc = embeddings_df[embeddings_df["model"] == "ArcFace"]
    gallery_base = arc[arc["split"] == "Gallery"].copy()

    all_ids = set(arc["base_identity"].unique())
    have_ids = set(gallery_base["base_identity"].unique())
    missing = sorted(all_ids - have_ids - {"Unknown"})

    seed_frames = [gallery_base]
    seeded_filenames: set[str] = set()

    for identity in missing:
        candidate = arc[(arc["base_identity"] == identity) & (arc["condition"] == "clean_probe")]
        if len(candidate) == 0:
            candidate = arc[arc["base_identity"] == identity].head(10)
            if verbose:
                print(f"  [{identity}] no clean_probe; seeding from {len(candidate)} probe rows")
        else:
            if verbose:
                print(f"  [{identity}] seeded with {len(candidate)} clean_probe rows")
        seeded_filenames.update(candidate["filename"].values)
        seed_frames.append(candidate)

    gallery = pd.concat(seed_frames, ignore_index=True)
    embeddings = np.vstack(gallery["embedding"].values)
    labels = gallery["base_identity"].values
    return embeddings, labels, seeded_filenames


def identify(unknown_vector: np.ndarray, gallery_emb: np.ndarray, gallery_lbl: np.ndarray, top_k: int = 3, verbose: bool = False) -> str:
    """Return the predicted identity for one ArcFace vector via cosine similarity."""
    vec = np.asarray(unknown_vector).reshape(1, -1)
    sims = cosine_similarity(vec, gallery_emb)[0]
    top_idx = sims.argsort()[::-1][:top_k]
    matches = [(gallery_lbl[i], float(sims[i])) for i in top_idx]
    if verbose:
        print(f"Top-{top_k} matches:")
        for rank, (name, score) in enumerate(matches, 1):
            flag = " <- predicted" if rank == 1 else ""
            print(f"  {rank}. {name:<22} similarity={score:.4f}{flag}")
    return matches[0][0]


def save_gallery(embeddings: np.ndarray, labels: np.ndarray, path: Path = GALLERY_PKL) -> None:
    with open(path, "wb") as f:
        pickle.dump({"embeddings": embeddings, "labels": labels}, f)


def load_gallery(path: Path = GALLERY_PKL) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["embeddings"], obj["labels"]


def append_to_gallery(new_embeddings: np.ndarray, new_labels: np.ndarray,
                      path: Path = GALLERY_PKL) -> tuple[np.ndarray, np.ndarray]:
    """Load current gallery, append new vectors, save. Returns the merged pair."""
    if path.exists():
        emb, lbl = load_gallery(path)
        emb = np.vstack([emb, new_embeddings])
        lbl = np.concatenate([lbl, new_labels])
    else:
        emb, lbl = new_embeddings, new_labels
    save_gallery(emb, lbl, path)
    return emb, lbl
