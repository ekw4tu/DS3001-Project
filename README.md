# Facial Recognition Final Project — Stage 2 + Stage 3

End-to-end facial-recognition pipeline built around two feature extractors
(**ArcFace** and **VGG19**) and the supervised models that sit on top of them.
The original Stage 2 and Stage 3 Colab notebooks are preserved verbatim under
[`notebooks/`](notebooks/); the same logic is also factored into reusable
Python modules under [`src/`](src/) and command-line scripts under
[`scripts/`](scripts/) so the whole thing runs from a clean clone with one or
two commands — on a laptop or on Colab with a GPU.

---

## How to run (step-by-step)

The pipeline is the same on a local machine and in Colab. Pick whichever
matches your setup.

### Local (CPU or CUDA)

```bash
# 1. Clone the repo
git clone https://github.com/ekw4tu/DS3001-Final.git
cd DS3001-Final

# 2. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 3. Drop the dataset under data/
#    Layout:  data/Gallery/<personGallery>/...
#             data/Probe/<personProbe>/<condition>/...
#    (Same shape as the original DS_NN_Project_Pictures folder.)

# 4. Extract features and build the gallery (replaces Stage 2)
python scripts/build_gallery.py            # CPU
python scripts/build_gallery.py --gpu      # CUDA, if available

# 5. Reproduce every Stage 3 metric on the held-out test set
python scripts/evaluate.py

# 6. Identify ArcFace unknown vectors (the in-class challenge)
python scripts/identify_unknowns.py path/to/unknown.npy --top-k 5

# 7. Identify VGG19 1000-d logits (also from the in-class challenge)
python scripts/robust_match_vgg_logits.py path/to/unknown.npy --classmates-only --top-k 5
```

### Colab (GPU — recommended for Challenge Day)

Open [`notebooks/Challenge_Day.ipynb`](notebooks/Challenge_Day.ipynb) in
Colab and run top to bottom:

1. **Runtime → Change runtime type → GPU (T4)**
2. Cell 1 clones the repo, pip-installs requirements, swaps onnxruntime for
   the GPU build.
3. Cell 2 mounts Drive and points `FR_DATA_ROOT` /
   `FR_ARTIFACTS_DIR` at the shared picture folder + artifacts cache.
4. Cell 3 runs `build_gallery.py --gpu`.
5. Cell 4 sniffs the dimensionality of every `*.npy` you uploaded so you know
   which matcher to call.
6. Cells 5–6 run the matchers — `identify_unknowns.py` for the 512-d ArcFace
   vectors and `robust_match_vgg_logits.py` for the 1000-d VGG19 logits.

The notebook is idempotent: re-running it on a fresh runtime reproduces every
result.

### Adding a new identity (no full rebuild)

```bash
mkdir -p data/Gallery/aliceGallery
cp /wherever/alice_*.jpg data/Gallery/aliceGallery/

python scripts/add_identity.py \
    --name "Alice Smith" \
    --folder data/Gallery/aliceGallery \
    --match-tokens alice smith
```

`add_identity.py` extracts ArcFace vectors only for Alice's folder and
appends them to `artifacts/gallery_embeddings.pkl`. It does not re-extract
the other 500+ images, so the loop is fast.

---

## Why this repo exists

The Stage 2 and Stage 3 notebooks ran on Colab against hardcoded Drive paths.
That's fine for the team but makes replication painful for anyone outside it.
This repo solves three problems the notebooks didn't:

1. **Replicability.** One `git clone` + `pip install` + one script. No Drive,
   no Colab runtime, no manual cell order.
2. **Extensibility.** Adding a new identity is one command
   (`scripts/add_identity.py`) instead of rerunning the full ArcFace loop.
3. **Portability.** Same code runs on a laptop CPU, on a Mac, or on Colab
   GPU. Paths are resolved by [`src/config.py`](src/config.py), not hardcoded.

The notebooks were not deleted. Every classifier, every metric, every plot
still lives in `notebooks/` for reference.

---

## Which model is the production identifier?

The best identity classifier on the held-out probe set is **ArcFace + cosine
similarity against the auto-seeded gallery** — 100% in Stage 3 cell 9. The
VGG19 paths (LR on raw features, L2+PCA+LR, MLP head, SupCon contrastive
head) remain in the pipeline as comparison baselines, and the trained head
weights are saved under `artifacts/`.

| Classifier | Task | Held-out accuracy |
|---|---|---|
| ArcFace + cosine (`identify()`) | Identity | **1.000** |
| ArcFace LR | Identity | 1.000 |
| VGG19 LR (raw) | Identity | 0.810 |
| VGG19 L2 + PCA + LR | Identity | 0.785 |
| VGG19 MLP head | Identity | 0.905 |
| VGG19 SupCon + nearest-centroid | Identity | reported by `evaluate.py` step [7] |
| ArcFace LR | Condition | reported by `evaluate.py` |
| VGG19 LR | Condition | reported by `evaluate.py` |

**Why ArcFace wins:** integrated face detection + 5-landmark affine alignment
(every face is warped to a canonical 112×112 pose before embedding) and
end-to-end identity training with additive angular margin loss. VGG19 is a
general-purpose ImageNet classifier with no alignment and no
identity-specific objective.

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
│   ├── train.py               # all Stage 3 classifiers
│   └── types.py               # shared TypedDict
├── scripts/                   # CLI entry points
│   ├── build_gallery.py       # one-shot: extract everything, split, save
│   ├── add_identity.py        # drop photos -> append to gallery
│   ├── evaluate.py            # reproduce Stage 3 metrics on held-out set
│   ├── identify_unknowns.py   # match 512-d ArcFace vectors
│   ├── match_vgg_logits.py    # match 1000-d VGG19 logits (clean gallery)
│   ├── robust_match_vgg_logits.py  # robust variant: trains on all images
│   ├── identify_vgg_clusters.py    # PCA-whitened VGG matcher (probe study)
│   └── robust_identify.py     # LR over the full embeddings_df, multi-model
├── notebooks/
│   ├── Stage2_Feature_Extraction.ipynb       # original Stage 2
│   ├── Stage2_Feature_Extraction_original.ipynb  # original Stage 2 backup
│   ├── Stage3_Supervised_Integration.ipynb   # original Stage 3
│   └── Challenge_Day.ipynb    # in-class identification driver
├── data/
│   ├── Gallery/<person>Gallery/*.jpg
│   └── Probe/<person>Probe/<condition>/*.jpg
├── artifacts/                 # generated: embeddings, gallery, model heads
└── tests/                     # smoke tests
```

---

## Data layout

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
[`src/metadata.py`](src/metadata.py)) is picked up automatically. Numeric
folder stems (`1Gallery`, `38Gallery`, …) are also recognized — that's how
the in-class classmate roster is loaded without 38 registry edits.

Condition is parsed from folder/filename substrings (`expression`,
`occlusion`/`glass`, `light`, `side`/`profile`).

---

## Walking through the pipeline

### Stage 2 — feature extraction and clustering

1. **Walk images** (`walk_image_tasks`). One pass over the filesystem so
   ArcFace and VGG19 see the *same* set of files — required for an
   apples-to-apples comparison later.
2. **Detect + align + embed with ArcFace** (`extract_arcface`). The
   InsightFace `get()` call runs RetinaFace (detection), affine alignment
   via 5 landmarks, then the ResNet-50 ArcFace head, returning a 512-d
   pose-normalized vector.
3. **Crop with InsightFace, embed with VGG19** (`extract_vgg19`). VGG19
   has no face detector. Reusing the ArcFace bounding box keeps the two
   feature sets defined on the same pixels.
4. **Metadata parsing** (`parse_base_identity`, `parse_condition`). Pulls
   identity and condition from folder + filename substrings; data-driven so
   new identities don't require editing code.
5. **k-means + Hungarian** (`run_kmeans`, `cluster_accuracy`). k-means
   assigns arbitrary cluster IDs, so we compute the optimal one-to-one
   permutation between clusters and identities before measuring accuracy.

### Stage 3 — supervised models + the production identifier

6. **Stratified 80/20 split** (build_gallery.py). Stratification on
   `identity + condition` ensures each cell of the grid (e.g. "Jack under
   occlusion") is present in both splits.
7. **Condition LR** (`train_condition_classifier`). Logistic regression for
   the noise condition. Class-balanced because Lighting has roughly 2× the
   samples of any other condition.
8. **Identity LR** (`train_identity_classifier`). Direct baseline for
   identity classification without any distance-based reasoning.
9. **VGG L2 + PCA + LR** (`train_vgg_pca_pipeline`). The lightweight
   "another logistic regression on top of VGG" path. L2 makes Euclidean
   distance approximate cosine; PCA-256 throws away noise dimensions.
10. **VGG MLP head** (`train_vgg_mlp_head`). The heavier finetune path —
    VGG weights are frozen; we train a 3-layer head with batch norm +
    dropout. The dataset is too small to update 140M parameters without
    catastrophic overfitting.
11. **VGG SupCon contrastive head** (`train_vgg_contrastive_head`). Trains
    a projection head with supervised contrastive loss (Khosla et al.
    2020) so that same-identity vectors cluster on the unit sphere.
    Inference is nearest-centroid, mirroring the ArcFace `identify()` path
    over a learned VGG-specific metric space.
    Architecture:
    `VGG19 (frozen) -> L2 -> Linear(4096, 512) -> BN -> ReLU -> Linear(512, 128) -> L2`.
12. **ArcFace gallery + `identify()`** (`build_gallery`, `identify`). The
    production classifier — cosine similarity over a gallery. The
    auto-seed-missing-identity trick means any identity present only under
    `Probe/` still ends up in the gallery, without a single edit to the
    loop.

---

## Environment variables

| Variable | Effect |
|---|---|
| `FR_PROJECT_ROOT` | Override repo root (normally auto-detected) |
| `FR_DATA_ROOT` | Override data path (Drive: `/content/drive/MyDrive/DS_NN_Project_Pictures`) |
| `FR_ARTIFACTS_DIR` | Where to save `.pkl`/`.pth` outputs |

---

## Notes

- `insightface` downloads the `buffalo_l` ONNX bundle (~280 MB) on first
  run. First run needs internet; subsequent runs hit the local cache at
  `~/.insightface/models/`.
- TensorFlow and PyTorch both ship in `requirements.txt` — TF because
  VGG19 comes from `tf.keras.applications`, PyTorch because the MLP and
  SupCon heads are trained in PyTorch.
- The test split uses `random_state=42`, so the held-out numbers are
  deterministic across machines.
