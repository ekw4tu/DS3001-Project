"""Environment-aware paths and constants.

Auto-detects Colab vs local so the pipeline runs in both without edits.
"""
from pathlib import Path
import os
import sys


def _is_colab() -> bool:
    return "google.colab" in sys.modules


def _project_root() -> Path:
    env = os.environ.get("FR_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    if _is_colab():
        drive_root = Path("/content/drive/MyDrive/DS_NN_Project_Pictures")
        if drive_root.exists():
            return drive_root
        return Path("/content/facial-recognition-project")
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = _project_root()
DATA_ROOT = Path(os.environ.get("FR_DATA_ROOT", PROJECT_ROOT / "data"))
GALLERY_DIR = DATA_ROOT / "Gallery"
PROBE_DIR = DATA_ROOT / "Probe"
ARTIFACTS_DIR = Path(os.environ.get("FR_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))

EMBEDDINGS_PKL = ARTIFACTS_DIR / "stage2_embeddings.pkl"
TRAIN_PKL = ARTIFACTS_DIR / "train_embeddings.pkl"
TEST_PKL = ARTIFACTS_DIR / "test_embeddings.pkl"
GALLERY_PKL = ARTIFACTS_DIR / "gallery_embeddings.pkl"
VGG_HEAD_PTH = ARTIFACTS_DIR / "vgg19_finetuned_head.pth"

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg")
K_IDENTITIES = 10
RANDOM_SEED = 42
ARCFACE_DIM = 512
VGG19_DIM = 4096

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
GALLERY_DIR.mkdir(parents=True, exist_ok=True)
PROBE_DIR.mkdir(parents=True, exist_ok=True)
