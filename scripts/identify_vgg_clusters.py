"""Stronger VGG19 cluster matching for the in-class challenge.

WHY this exists: raw cosine on VGG19 fc2 features is dominated by magnitude and
by shared ImageNet texture, so max-sim over a thin per-cluster gallery gave top
similarities around 0.27–0.41 — barely above noise, and the correct cluster
didn't even land in the top-5. This module applies the standard fixes for weak
embeddings (L2-norm, mean-centering, PCA-whitening) and swaps max-sim for
top-k-mean aggregation which is less sensitive to one noisy cluster member.

Designed to be importable from a Colab cell:

    from identify_vgg_clusters import identify_probe
    identify_probe(probe_path, gallery_vecs, gallery_clusters, top_k=5)
"""
from __future__ import annotations

import numpy as np


def _l2(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, eps, None)


def prepare_gallery(
    gallery_vecs: np.ndarray,
    pca_dim: int | None = 256,
    whiten: bool = True,
):
    """Return (processed_gallery, transform_fn).

    transform_fn applies the SAME mean-center + PCA-whiten to probe vectors.
    Both outputs are L2-normalized so cosine == dot product.
    """
    X = np.asarray(gallery_vecs, dtype=np.float64)
    X = _l2(X)
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
        v = _l2(v)
        v = v - mean
        v = (v @ components.T) * scale
        return _l2(v)

    return Xp, transform


def _aggregate(sims: np.ndarray, labels: np.ndarray, how: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for c in np.unique(labels):
        s = sims[labels == c]
        if how == "max":
            out[int(c)] = float(s.max())
        elif how == "mean":
            out[int(c)] = float(s.mean())
        elif how == "top2mean":
            k = min(2, len(s))
            out[int(c)] = float(np.sort(s)[::-1][:k].mean())
        elif how == "top3mean":
            k = min(3, len(s))
            out[int(c)] = float(np.sort(s)[::-1][:k].mean())
        else:
            raise ValueError(how)
    return out


def score_probe(
    probe_vecs: np.ndarray,
    gallery_proc: np.ndarray,
    gallery_clusters: np.ndarray,
    transform,
    strategies: tuple[str, ...] = ("max", "mean", "top2mean"),
) -> dict[str, list[tuple[int, float]]]:
    """Return per-strategy ranked cluster list for one probe (averaged over its
    vectors if it has multiple)."""
    P = transform(probe_vecs)
    sims_per_vec = P @ gallery_proc.T
    sims = sims_per_vec.mean(axis=0)

    ranked: dict[str, list[tuple[int, float]]] = {}
    for how in strategies:
        agg = _aggregate(sims, np.asarray(gallery_clusters), how)
        ranked[how] = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)

    fused: dict[int, float] = {}
    for how, lst in ranked.items():
        ranks = {c: r for r, (c, _) in enumerate(lst)}
        for c in ranks:
            fused[c] = fused.get(c, 0.0) + 1.0 / (60.0 + ranks[c])
    ranked["fusion"] = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return ranked


def identify_probe(
    probe_vecs: np.ndarray,
    gallery_vecs: np.ndarray,
    gallery_clusters: np.ndarray,
    top_k: int = 5,
    pca_dim: int | None = 256,
    whiten: bool = True,
    strategies: tuple[str, ...] = ("max", "mean", "top2mean"),
    label: str = "",
) -> dict[str, list[tuple[int, float]]]:
    """Preprocess gallery, score probe, pretty-print per-strategy top-k."""
    gal, tf = prepare_gallery(gallery_vecs, pca_dim=pca_dim, whiten=whiten)
    ranked = score_probe(
        np.atleast_2d(probe_vecs), gal, np.asarray(gallery_clusters), tf, strategies
    )
    header = f"=== {label} ===" if label else "==="
    print(header)
    all_strategies = list(strategies) + ["fusion"]
    width = max(len(s) for s in all_strategies)
    for how in all_strategies:
        print(f"  [{how:<{width}}]  " + "  ".join(
            f"{c}({s:.3f})" for c, s in ranked[how][:top_k]
        ))
    return ranked


if __name__ == "__main__":
    import argparse
    import pickle
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", required=True,
                    help="pkl/npz with 'vectors' and 'clusters', or two .npy via --gallery-vecs/--clusters")
    ap.add_argument("--probes", nargs="+", required=True, help=".npy files of probe vectors")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--pca-dim", type=int, default=256)
    ap.add_argument("--no-whiten", action="store_true")
    args = ap.parse_args()

    path = Path(args.gallery)
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        vecs, clusters = obj["vectors"], obj["clusters"]
    else:
        obj = np.load(path)
        vecs, clusters = obj["vectors"], obj["clusters"]

    for p in args.probes:
        pv = np.load(p)
        identify_probe(
            pv, vecs, clusters,
            top_k=args.top_k,
            pca_dim=args.pca_dim,
            whiten=not args.no_whiten,
            label=p,
        )
