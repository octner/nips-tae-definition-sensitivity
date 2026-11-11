"""This module defines the folders, constants and hash functions that the
other scripts import.

The function check_guard reads and writes the sidecar file of each results
file. README.md describes the expected contents of the folders that are not
released.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Released
REGISTRY = ROOT / "registry" / "prompt-registry.json"  # one record per prompt
TABLES = ROOT / "tables"                                # the analysis tables

# Not released
DATA = ROOT / "data"        # sample.parquet, sample_expansion.parquet, corrupt_fetch_ids.csv
RAW = DATA / "raw"          # the public datasets that sample_expansion.py draws from
DEFS = ROOT / "defs"        # the prompt files named in the registry's `file` field
OUT = ROOT / "out"
LABELS = OUT / "labels"                          # <model>__<construct>__<def_id>__<impl_id>.jsonl
LABELS_SECOND_PLATFORM = OUT / "labels_second_platform"  # the re-run on a second platform
LABELS_QWEN_FIRST_RUN = OUT / "labels_qwen_first_run"    # Qwen's first run on the other four concepts
COMPREHENSION = OUT / "comprehension_check"      # g2_<model>.csv, the answers in the comprehension check
SONNET_REPEAT = OUT / "sonnet_repeat"            # Sonnet's repeat batches
SONNET_BATCHES = OUT / "sonnet_batches"          # the texts of each Sonnet batch
FIGS = ROOT / "figs"        # the figures, written by an_figures.py
PAPER = ROOT / "paper"      # numbers.tex and the LaTeX tables, written by an_numbers.py

MODELS = ("gemma4_31b", "qwen36_27b", "claude_sonnet5")
CONCEPTS = ("protest", "assassination", "hate_speech", "climate_contrarian",
            "fake_news", "terrorism")
VERDICTS = ("INCLUDE", "EXCLUDE", "INDETERMINATE")

# The version of the prompt set is part of the hash of each prompt. The
# prompts of the other four concepts, LATER_CONCEPTS, have version tae-0.2,
# because we pre-registered these concepts after protest and assassination.
LATER_CONCEPTS = {"terrorism", "hate_speech", "climate_contrarian", "fake_news"}


def suite_version_for(construct: str) -> str:
    return "tae-0.2" if construct in LATER_CONCEPTS else "tae-0.1"


def sha(*parts: str, n: int = 12) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:n]


def instrument_sha(construct: str, def_id: str, impl_id: str, rendered: str) -> str:
    """The hash of one prompt, as the registry records it."""
    return sha(suite_version_for(construct), construct, def_id, impl_id, rendered)


def frame_sha_of(units, n: int = 12) -> str:
    """The hash of a sample, based on the doc_id, character span and word
    count of each text."""
    key = units[["doc_id", "char_span", "n_words"]].astype(str)
    payload = "\n".join(sorted("\t".join(r) for r in key.itertuples(index=False)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:n]


def check_guard(path: Path, instrument: str, frame: str) -> None:
    """Stop the run if a sidecar file records another prompt or sample.

    The sidecar of each results file contains the hash of the prompt, the
    hash of the sample and the version of the prompt set. If the sidecar is
    absent, the function writes it.
    """
    side = path.with_suffix(".guard.json")
    construct = path.stem.split("__")[1] if "__" in path.stem else ""
    want = {"instrument_sha": instrument, "frame_sha": frame,
            "suite_version": suite_version_for(construct)}
    if not side.exists():
        side.write_text(json.dumps(want, indent=2))
        return
    have = json.loads(side.read_text())
    if have != want:
        raise SystemExit(
            f"[tae] {path.name} was written with other hashes.\n"
            f"       on disk: {have}\n"
            f"       current: {want}\n"
            f"       Delete the results file to label the texts again.")
