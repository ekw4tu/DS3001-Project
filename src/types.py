"""Shared type definitions."""
from typing import TypedDict

import numpy as np


class EmbeddingRecord(TypedDict):
    filename: str
    identity: str
    base_identity: str
    condition: str
    split: str
    model: str
    embedding: np.ndarray
