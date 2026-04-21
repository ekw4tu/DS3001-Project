# Facial Recognition Final Project — Stage 2 + Stage 3

Reproducible, TA-friendly packaging of our Data Science Pics facial recognition
pipeline. The original Colab notebooks (Stage 2 and Stage 3) are preserved
verbatim under [`notebooks/`](notebooks/); the same code is also lifted into
reusable Python modules under [`src/`](src/) so the pipeline runs end-to-end
from a single command — on a laptop *or* on Colab with GPU.

---

## TL;DR — for a grader

```bash
git clone <REPO_URL>
cd "Data Science Pics"
pip install -r requirements.txt

# 1. Drop images under data/Gallery/<personGallery>/... and data/Probe/<personProbe>/<condition>/...
#    (Same folder layout as the original DS_NN_Project_Pictures/ on Google Drive.)

# 2. Extract features + build the gallery (one command, replaces Stage 2)
python scripts/build_gallery.py

# 3. Reproduce every Stage 3 metric on the held-out test set
python scripts/evaluate.py

# 4. In-class challenge: identify ArcFace vectors supplied by the TA
python scripts/identify_unknowns.py unknowns.npy --top-k 5
```

GPU (faster, optional):

```bash
python scripts/build_gallery.py --gpu
```

Colab path: open [`notebooks/Colab_Quickstart.ipynb`](notebooks/Colab_Quickstart.ipynb)
in Colab, run top to bottom. It clones this repo, mounts Drive, and calls the
same scripts — you get GPU for free.

---

## Why this exists

The Stage 2 and Stage 3 notebooks ran on Colab with hardcoded paths like
`/content/drive/MyDrive/DS_NN_Project_Pictures/`. That's fine for us but makes
replication painful: a TA or a collaborator would need to replicate our Drive
layout, copy our notebook, and re-run every cell.

This repo solves three problems the original notebooks did not:

1. **Replicability.** One `git clone` + `pip install` + one script. No Drive,
   no Colab runtime, no manual cell order.
2. **Extensibility.** Adding a new identity is one command
   (`scripts/add_identity.py`) instead of rerunning the full ArcFace loop.
3. **Portability.** Same code runs on a laptop CPU, a Mac, or Colab GPU. Paths
   are resolved by [`src/config.py`](src/config.py), not hardcoded.

Nothing from the notebooks was deleted. Every classifier, every metric, every
plot the graders will read still lives in `notebooks/`.

---

## Which model is "the optimized one"?

Our best identity classifier on the held-out probe set is **ArcFace + cosine
similarity against the auto-seeded gallery** — it scored **100%** in Stage 3
cell 9. The VGG19 fine-tuning the PDF asks for (LR on raw features, L2+PCA+LR,
and the MLP head) is still in the pipeline as a **comparison baseline** and the
MLP head artifact is saved under `artifacts/vgg19_finetuned_head.pth`.

| Classifier | Task | Held-out accuracy |
|---|---|---|
| ArcFace + cosine (`identify()`) | Identity | **1.000** |
| ArcFace LR | Identity | 1.000 |
| VGG19 LR (raw) | Identity | 0.810 |
| VGG19 L2 + PCA + LR | Identity | 0.785 |
| VGG19 MLP head (ours) | Identity | 0.905 |
| VGG19 SupCon + nearest-centroid (ours) | Identity | see `evaluate.py` step [7] |
| ArcFace LR | Condition | see `evaluate.py` |
| VGG19 LR | Condition | see `evaluate.py` |

**Why ArcFace wins:** it has integrated detection + alignment (warps every face
to a canonical 112×112 pose before embedding) and was trained end-to-end on
face identity with additive angular margin loss. VGG19 is a general-purpose
ImageNet classifier with no alignment and no identity-specific training.

---

## Repository layout

```
.
├── README.md                  # you are here
├── requirements.txt           # pinned deps
├── src/                       # library code (importable)
│   ├── config.py              # path + constant resolution (Colab/local aware)
│   ├── metadata.py            # identity + condition parsing, registry
│   ├── feature_extraction.py  # ArcFace + VGG19 extractors
│   ├── clustering.py          # Stage 2 k-means + Hungarian accuracy
│   ├── identify.py            # gallery build + identify()
│   └── train.py               # all Stage 3 classifiers
├── scripts/                   # CLI entry points
│   ├── build_gallery.py       # one-shot: extract everything, split, save
│   ├── add_identity.py        # drop photos -> append to gallery
│   ├── evaluate.py            # reproduce Stage 3 metrics on held-out set
│   └── identify_unknowns.py   # in-class challenge runner
├── notebooks/
│   ├── Stage2_Feature_Extraction.ipynb  # original Stage 2 (unchanged)
│   ├── Stage3_Supervised_Integration.ipynb  # original Stage 3 (unchanged)
│   └── Colab_Quickstart.ipynb # one-cell Colab bootstrap
├── data/
│   ├── Gallery/<person>Gallery/*.jpg
│   └── Probe/<person>Probe/<condition>/*.jpg
└── artifacts/                 # generated: embeddings, gallery, model heads
```

---

## Data layout (must match the original convention)

```
data/
├── Gallery/
│   ├── willGallery/          willSmithGallery1.jpg, willSmithGallery2.jpg, ...
│   ├── kimGallery/           ...
│   └── <personGallery>/      ...
└── Probe/
    ├── willProbe/
    │   ├── willExpression/   ...
    │   ├── willOcclusion/    ...
    │   ├── willLighting/     ...
    │   └── willSide/         ...
    ├── kimProbe/...
    └── <personProbe>/<condition>/*
```

Any folder whose name contains a known identity token (see
[`src/metadata.py`](src/metadata.py)) is picked up automatically. Condition is
parsed from folder/filename substrings (`expression`, `occlusion`/`glass`,
`light`, `side`/`profile`).

---

## How to add a new identity

The whole point of this refactor. Three steps:

```bash
# 1. Drop clean frontal photos of the new person into data/Gallery/aliceGallery/
mkdir -p data/Gallery/aliceGallery
cp /wherever/alice_*.jpg data/Gallery/aliceGallery/

# 2. Run add_identity. --name is the canonical label, --match-tokens optional
python scripts/add_identity.py \
    --name "Alice Smith" \
    --folder data/Gallery/aliceGallery \
    --match-tokens alice smith

# 3. (Optional) drop probe images too, then rerun build_gallery to include them
#    in train/test splits and per-condition accuracy reports.
python scripts/build_gallery.py
```

**Why this is fast:** `add_identity.py` only extracts ArcFace vectors for the
new folder (seconds) and appends them to `artifacts/gallery_embeddings.pkl`. It
does not rerun extraction for the other 500+ images.

**Why the tokens matter:** they tell the metadata parser which filenames belong
to Alice in future probe sets. If you name every Alice file `alice_<n>.jpg`,
`--match-tokens alice` is enough. The tokens persist in
`src/identities.json` so they survive across runs.

---

## Requirements checklist — Stage 2 PDF

| PDF requirement | Where it lives |
|---|---|
| Concise architecture description of ArcFace + VGG19 | [`notebooks/Stage2_Feature_Extraction.ipynb`](notebooks/Stage2_Feature_Extraction.ipynb) markdown cells 19, 20 |
| ArcFace -> 512-d embedding, VGG19 -> 4096-d | [`src/feature_extraction.py`](src/feature_extraction.py) `extract_arcface`, `extract_vgg19` |
| InsightFace `get()` pipeline explanation (detection + alignment + embedding) | Stage 2 notebook markdown cell 20 |
| ONNX graph inspection (`onnx.helper.printable_graph`) | Stage 2 notebook (cell referencing `w600k_r50.onnx`) |
| k-means implementation | [`src/clustering.py`](src/clustering.py) `run_kmeans` |
| k-means on clean (10 identities) + Hungarian-aligned accuracy | Stage 2 cell 35 |
| ArcFace vs VGG19 comparison on clean | Stage 2 cell 35 (97% vs 62%) |
| k-means on noisy data per condition | Stage 2 cell 33 |
| Robustness discussion, ArcFace vs VGG19 | Stage 2 markdown cell 36 |
| Feature distribution for 2 identities across conditions | Stage 2 cells 30–31 (PCA scatter) |

## Requirements checklist — Stage 3 PDF

| PDF requirement | Where it lives |
|---|---|
| Logistic regression for condition classification | [`src/train.py`](src/train.py) `train_condition_classifier` |
| Use ArcFace and VGG19 features | Covered by `--feature-model` flag on train; evaluated in `scripts/evaluate.py` steps [1] and [2] |
| Performance across conditions, held-out test set only | `scripts/evaluate.py` step [8] (per-condition breakdown) |
| Fine-tune re-identification on VGG features (LR-on-top) | `src/train.py` `train_vgg_pca_pipeline` |
| Fine-tune VGG layers (MLP head over frozen VGG) | `src/train.py` `train_vgg_mlp_head` |
| Fine-tune VGG for clustering / distance-based id | `src/train.py` `train_vgg_contrastive_head` (SupCon projection head + nearest-centroid cosine) |
| Documentation of full pipeline | this README + docstrings in `src/` |
| Instructions for running on held-out test sets | `scripts/evaluate.py` is the one command |
| In-class challenge: match 5 unknown feature vectors | `scripts/identify_unknowns.py` |

---

## Walking through the pipeline (the WHY of each stage)

### Stage 2 — feature extraction and clustering

1. **Walk images** (`walk_image_tasks`). WHY: one pass over the filesystem so
   ArcFace and VGG19 see the *same* set of files, which is a prerequisite for
   apples-to-apples comparison later.
2. **Detect + align + embed with ArcFace** (`extract_arcface`). WHY: the
   InsightFace `get()` call runs RetinaFace (detection), affine alignment
   (via 5 landmarks), then the ResNet-50 ArcFace head — one call gives us
   the 512-d vector that's already been normalized for pose.
3. **Crop with InsightFace, embed with VGG19** (`extract_vgg19`). WHY: VGG19
   has no face detector. Reusing the ArcFace bounding box means the two
   feature sets are compared on the same pixels.
4. **Metadata parsing** (`parse_base_identity`, `parse_condition`). WHY:
   pulls the canonical identity label and noise condition from folder +
   filename substrings, and it's data-driven so new identities don't require
   editing code.
5. **k-means + Hungarian** (`run_kmeans`, `cluster_accuracy`). WHY: k-means
   gives arbitrary cluster IDs, so we compute the best one-to-one mapping
   between clusters and identities before measuring accuracy.

### Stage 3 — supervised models + the production identifier

6. **Stratified 80/20 split** (build_gallery.py). WHY: stratification on
   `identity + condition` ensures each cell of the grid (e.g. "Jack under
   occlusion") lands in both splits — otherwise a lucky random draw could
   leave an entire condition untested.
7. **Condition LR** (`train_condition_classifier`). WHY: answers "what
   condition is this image in" on held-out data. Class-balanced because the
   condition distribution is skewed (Lighting is ~2x any other).
8. **Identity LR** (`train_identity_classifier`). WHY: direct baseline for
   identity classification without any distance-based reasoning.
9. **VGG L2 + PCA + LR** (`train_vgg_pca_pipeline`). WHY: the lightweight
   "another logistic regression on top of VGG" path from the PDF. L2 makes
   euclidean distance approximate cosine; PCA to 256 throws away noise.
10. **VGG MLP head** (`train_vgg_mlp_head`). WHY: the heavier "finetune VGG"
    path from the PDF. We keep VGG weights frozen and train a 3-layer head
    with batch norm + dropout. Dataset is too small to finetune 140M params
    without overfitting.
11. **VGG SupCon contrastive head** (`train_vgg_contrastive_head`). WHY: the
    Stage 3 PDF asks us to finetune VGG so its features are strong enough
    for *clustering/distance-based identification*. Cross-entropy on a
    closed label set does not shape the metric space — it only separates
    classes at the decision boundary. SupCon (Khosla et al. 2020) directly
    pulls same-identity embeddings together on the unit sphere and pushes
    different-identity embeddings apart, which is exactly the geometry a
    cosine-NN / k-means lookup exploits. Architecture:
    `VGG19 (frozen) -> L2 -> Linear(4096, 512) -> BN -> ReLU -> Linear(512, 128) -> L2`.
    Inference is nearest-centroid on the projected unit sphere.
12. **ArcFace gallery + `identify()`** (`build_gallery`, `identify`). WHY:
    this is the production classifier — cosine similarity over a gallery.
    The auto-seed-missing-identity trick (from Stage 3 cell 9) means any
    identity present in Probe/ but missing from Gallery/ still ends up in
    the gallery, without a single edit to the loop.

---

## Environment variables (optional)

| Variable | Effect |
|---|---|
| `FR_PROJECT_ROOT` | Override repo root (normally auto-detected) |
| `FR_DATA_ROOT` | Override data path (use for Drive: `/content/drive/MyDrive/DS_NN_Project_Pictures`) |
| `FR_ARTIFACTS_DIR` | Where to save `.pkl`/`.pth` outputs |

---

## Deadlines

- **4/29** for the 110% early-submission multiplier (Stage 3 §2).
- **5/6** regular deadline.
- **4/21 and 4/23** in-class unknown-identity challenges — run
  `scripts/identify_unknowns.py` against whatever vectors the TA hands out.

---

## Known caveats

- `insightface` downloads `buffalo_l` ONNX weights (~280 MB) on first run.
  First run needs internet. Subsequent runs hit the local cache at
  `~/.insightface/models/`.
- TensorFlow and PyTorch both ship in `requirements.txt` — TF because VGG19
  comes from `tf.keras.applications`, PyTorch because the MLP head is
  trained in PyTorch. If install size is a concern on the grader's machine,
  `pip install -r requirements.txt --no-deps` followed by selective installs
  works, but the full install is the recommended path.
- The test split is produced with a fixed `random_state=42`, so the TA's
  held-out numbers will exactly match ours.
