"""Add a new identity to the gallery without rebuilding everything.

Usage:
  python scripts/add_identity.py --name "Alice Smith" --folder data/Gallery/aliceGallery
  python scripts/add_identity.py --name "Alice Smith" --folder data/Gallery/aliceGallery \
      --match-tokens alice smith

WHY a dedicated script: rebuilding the full gallery takes minutes because
ArcFace has to re-extract every image. Appending one person's vectors takes
seconds, which is what makes the "drop photos, get recognition" loop fast.
"""
from pathlib import Path
import argparse
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_extraction import init_arcface, extract_arcface_for_folder
from src.identify import append_to_gallery
from src.metadata import register_identity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help='Canonical display name, e.g. "Alice Smith"')
    ap.add_argument("--folder", required=True, type=Path,
                    help="Folder of images for this person (any of .jpg/.jpeg/.png)")
    ap.add_argument("--match-tokens", nargs="*", default=None,
                    help="Substrings that identify this person in future filenames. "
                         "Defaults to the lowercase first name.")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    if not args.folder.exists():
        sys.exit(f"Folder not found: {args.folder}")

    tokens = args.match_tokens or [args.name.split()[0].lower()]
    register_identity(args.name, tokens)
    print(f"Registered identity tokens: {args.name} -> {tokens}")

    print(f"Extracting ArcFace vectors from {args.folder} ...")
    app = init_arcface(use_gpu=args.gpu)
    records = extract_arcface_for_folder(args.folder, app, label=args.name)
    if not records:
        sys.exit("No faces detected. Check image quality / orientation.")

    embeddings = np.vstack([r["embedding"] for r in records])
    labels = np.array([r["base_identity"] for r in records])
    emb, lbl = append_to_gallery(embeddings, labels)
    print(f"Appended {len(labels)} vectors. Gallery now holds {len(lbl)} vectors across "
          f"{len(set(lbl))} identities.")


if __name__ == "__main__":
    main()
