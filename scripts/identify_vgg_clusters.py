"""Stronger VGG19 matching for the in-class challenge (1000-d logits).

WHY this exists: raw cosine on 1000-d VGG19 pre-softmax logits is dominated by
ImageNet class bias -- most of those dims have nothing to do with faces -- so
max-sim over a thin per-identity gallery landed the correct identity outside
the top-5 for ID4 and ID5_UPDATED. This module applies the standard fixes for
weak embeddings (softmax, mean-center, drop dead dims, PCA-whiten) and swaps
max-sim for top-k-mean aggregation which is less sensitive to a noisy member.

Colab usage (self-contained -- builds the gallery from the saved pkl):

    from identify_vgg_clusters import identify_from_pkl
    identify_from_pkl(
        pkl='/content/drive/MyDrive/facial_recognition_artifacts/stage2_embeddings.pkl',
        probes=['/content/ID4.npy', '/content/ID5_UPDATED.npy'],
    )
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def _l2(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, eps, None)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def prepare_gallery(
    gallery_vecs: np.ndarray,
    mode: str = "softmax",
    pca_dim: int | None = 128,
    whiten: bool = True,
    drop_low_var: float = 1e-4,
):
    """Return (processed_gallery, transform_fn, kept_dims).

    mode:
      'softmax' -- best for 1000-d logits: softmax across classes first so
                   magnitude stops dominating.
      'l2'      -- for generic features (e.g. 4096-d fc2).
    """
    X = np.asarray(gallery_vecs, dtype=np.float64)
    if mode == "softmax":
        X = _softmax(X, axis=-1)
    else:
        X = _l2(X)

    var = X.var(axis=0)
    kept = var > drop_low_var
    if kept.sum() < 16:
        kept = np.ones_like(kept, dtype=bool)
    X = X[:, kept]

    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean

    if pca_dim is not None and pca_dim < Xc.shape[1]:
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(pca_dim, Vt.shape[0])
        components = Vt[:k]
        scale = 1.0 / np.clip(S[:k], 1e-6, None) if whiten else np.ones(k)
        Xp = (Xc @ components.T) * scale
    else:
        components = np.eye(Xc.shape[1])
        scale = np.ones(Xc.shape[1])
        Xp = Xc

    Xp = _l2(Xp)

    def transform(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        if v.ndim == 1:
            v = v[None, :]
        v = _softmax(v, axis=-1) if mode == "softmax" else _l2(v)
        v = v[:, kept]
        v = v - mean
        v = (v @ components.T) * scale
        return _l2(v)

    return Xp, transform, kept


def _aggregate(sims: np.ndarray, labels: np.ndarray, how: str) -> dict:
    out: dict = {}
    for c in np.unique(labels):
        s = sims[labels == c]
        if how == "max":
            out[c] = float(s.max())
        elif how == "mean":
            out[c] = float(s.mean())
        elif how == "top2mean":
            k = min(2, len(s))
            out[c] = float(np.sort(s)[::-1][:k].mean())
        elif how == "top3mean":
            k = min(3, len(s))
            out[c] = float(np.sort(s)[::-1][:k].mean())
        else:
            raise ValueError(how)
    return out


def score_probe(
    probe_vecs: np.ndarray,
    gallery_proc: np.ndarray,
    gallery_labels: np.ndarray,
    transform,
    strategies=("max", "mean", "top2mean"),
):
    P = transform(probe_vecs)
    sims = (P @ gallery_proc.T).mean(axis=0)
    ranked = {}
    for how in strategies:
        agg = _aggregate(sims, np.asarray(gallery_labels), how)
        ranked[how] = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
    fused: dict = {}
    for how, lst in ranked.items():
        for r, (c, _) in enumerate(lst):
            fused[c] = fused.get(c, 0.0) + 1.0 / (60.0 + r)
    ranked["fusion"] = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return ranked


def _print_ranked(ranked, top_k, label):
    print(f"=== {label} ===" if label else "===")
    order = list(ranked)
    w = max(len(s) for s in order)
    for how in order:
        cells = "  ".join(f"{c!s:>15}({s:.3f})" for c, s in ranked[how][:top_k])
        print(f"  [{how:<{w}}]  {cells}")


def load_vgg_gallery(pkl_path):
    """Load stage2_embeddings.pkl, filter to VGG19 Gallery rows.

    Returns (vectors [N, D], labels [N] of base_identity strings).
    Falls back to clean_probe rows for any identity missing from Gallery.
    """
    with open(pkl_path, "rb") as f:
        df = pickle.load(f)
    vgg = df[df["model"] == "VGG19"]
    gal = vgg[vgg["split"] == "Gallery"]
    all_ids = set(vgg["base_identity"].unique()) - {"Unknown"}
    missing = all_ids - set(gal["base_identity"].unique())
    frames = [gal]
    for ident in sorted(missing):
        cand = vgg[(vgg["base_identity"] == ident) & (vgg["condition"] == "clean_probe")]
        if len(cand) == 0:
            cand = vgg[vgg["base_identity"] == ident].head(10)
        frames.append(cand)
    import pandas as pd
    merged = pd.concat(frames, ignore_index=True)
    vecs = np.vstack(merged["embedding"].values)
    labels = merged["base_identity"].values
    return vecs, labels


def identify_probe(
    probe_vecs,
    gallery_vecs,
    gallery_labels,
    top_k: int = 5,
    mode: str = "softmax",
    pca_dim: int | None = 128,
    whiten: bool = True,
    label: str = "",
):
    gal, tf, _ = prepare_gallery(gallery_vecs, mode=mode, pca_dim=pca_dim, whiten=whiten)
    ranked = score_probe(np.atleast_2d(probe_vecs), gal, gallery_labels, tf)
    _print_ranked(ranked, top_k, label)
    return ranked


def identify_from_pkl(
    pkl: str,
    probes: list,
    top_k: int = 5,
    mode: str = "softmax",
    pca_dim: int | None = 128,
):
    """Full pipeline: load VGG gallery from pkl, score each probe file."""
    vecs, labels = load_vgg_gallery(pkl)
    print(f"Loaded VGG gallery: {len(vecs)} vectors, {len(set(labels))} identities")
    gal, tf, kept = prepare_gallery(vecs, mode=mode, pca_dim=pca_dim, whiten=True)
    print(f"  kept {kept.sum()}/{len(kept)} dims after low-variance drop; PCA -> {gal.shape[1]}\n")
    results = {}
    for p in probes:
        pv = np.load(p)
        if pv.ndim == 1:
            pv = pv[None, :]
        ranked = score_probe(pv, gal, labels, tf)
        _print_ranked(ranked, top_k, p)
        results[p] = ranked
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--probes", nargs="+", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--mode", choices=["softmax", "l2"], default="softmax")
    ap.add_argument("--pca-dim", type=int, default=128)
    args = ap.parse_args()
    identify_from_pkl(args.pkl, args.probes, top_k=args.top_k,
                      mode=args.mode, pca_dim=args.pca_dim)
