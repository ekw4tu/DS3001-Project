"""Data-driven identity and condition parsing.

Tokens are matched against folder+filename substrings. `add_identity.py`
extends the registry at runtime via a JSON override next to this file.
"""
from pathlib import Path
import json

_OVERRIDE_FILE = Path(__file__).with_name("identities.json")

DEFAULT_IDENTITY_TOKENS = {
    "Will Smith":        ["will"],
    "Tom Cruise":        ["cruise"],
    "Nico":              ["nico"],
    "Jack":              ["jack"],
    "Kim Kardashian":    ["kim"],
    "Leonardo DiCaprio": ["dicaprio", "leo"],
    "Jalen Brunson":     ["brunson"],
    "Taylor Swift":      ["taylor", "swift"],
    "Megan Fox":         ["megan", "fox"],
    "Dylan":             ["dylan"],
}

DEFAULT_CONDITION_TOKENS = {
    "Expression": ["expression"],
    "Occlusion":  ["occlusion", "glass", "sunglasses"],
    "Lighting":   ["lighting", "light"],
    "Side":       ["side", "profile"],
}


def _load_overrides() -> dict:
    if _OVERRIDE_FILE.exists():
        with _OVERRIDE_FILE.open() as f:
            return json.load(f)
    return {}


def identity_tokens() -> dict[str, list[str]]:
    merged = dict(DEFAULT_IDENTITY_TOKENS)
    merged.update(_load_overrides().get("identities", {}))
    return merged


def condition_tokens() -> dict[str, list[str]]:
    merged = dict(DEFAULT_CONDITION_TOKENS)
    merged.update(_load_overrides().get("conditions", {}))
    return merged


def register_identity(canonical_name: str, tokens: list[str]) -> None:
    """Persist a new identity so parse_base_identity recognizes it."""
    overrides = _load_overrides()
    overrides.setdefault("identities", {})[canonical_name] = tokens
    _OVERRIDE_FILE.write_text(json.dumps(overrides, indent=2))


def parse_base_identity(folder_name: str, file_name: str = "") -> str:
    """Return canonical identity from folder + filename substrings."""
    s = (folder_name + file_name).lower()
    for name, tokens in identity_tokens().items():
        if any(tok in s for tok in tokens):
            return name
    return "Unknown"


def parse_condition(folder_name: str, file_name: str, split: str) -> str:
    """Return one of: clean, clean_probe, Expression, Occlusion, Lighting, Side."""
    if split.strip() == "Gallery":
        return "clean"
    s = (folder_name + file_name).lower()
    for cond, tokens in condition_tokens().items():
        if any(tok in s for tok in tokens):
            return cond
    return "clean_probe"


def merge_clean(cond: str) -> str:
    """Collapse clean + clean_probe into a single 'Clean' bucket for condition classifiers."""
    return "Clean" if cond in ("clean", "clean_probe") else cond
