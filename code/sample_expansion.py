"""Draw the texts for hate speech, climate contrarianism and fake news.

The script draws 600 texts at random for each of these three concepts from a
public dataset. Terrorism uses the assassination texts, so the script does not
draw texts for it.

    hate speech            Civil Comments (Borkan et al. 2019)
    climate contrarianism  CLIMATE-FEVER (Diggelmann et al. 2020)
    fake news              LIAR (Wang 2017)

In the code, a construct is a concept and a unit is a text. Each text also has
a block, a group of texts that the bootstrap of the analysis resamples
together. The loaders use only the text and the columns that define the
blocks, so the draw cannot depend on the annotations in the datasets. The
script draws only from texts of 8 to 400 words, without exact duplicates.

The script reads the datasets in data/raw/expansion/ and writes
data/sample_expansion.parquet, with the same columns as sample.parquet. The
frame_sha column contains the hash of the drawn texts, b60a8b7535a9.

Run from the repository root.
    python3 code/sample_expansion.py
"""
import hashlib
import json
import re
import sys

import pandas as pd

from _paths import DATA, RAW, frame_sha_of

SEED = 20260823
N_PER = 600
WORDS_MIN, WORDS_MAX = 8, 400
RAW_X = RAW / "expansion"


def _load_civil_comments() -> pd.DataFrame:
    """Civil Comments, the train files of google/civil_comments (CC0).

    The loader reads only the text column, without the toxicity scores.
    Without a thread identifier in the dataset, each text is its own block.
    """
    shards = sorted((RAW_X / "civil_comments" / "data").glob("train-*.parquet"))
    if not shards:
        sys.exit(f"missing {RAW_X}/civil_comments/data/train-*.parquet")
    df = pd.concat([pd.read_parquet(s, columns=["text"]) for s in shards],
                   ignore_index=True)
    df["block_id"] = ""      # draw() uses the doc_id as the block
    df["source_id"] = ""
    return df


def _load_climate_fever() -> pd.DataFrame:
    """CLIMATE-FEVER, the JSONL file of the University of Zurich.

    The loader uses only claim_id and claim, and each claim is its own block.
    """
    src = RAW_X / "climate-fever-dataset-r1.jsonl"
    if not src.exists():
        sys.exit(f"missing {src}. Download the JSONL file from "
                 "sustainablefinance.uzh.ch/en/research/climate-fever.html")
    rows = [(str(r["claim_id"]), r["claim"])
            for r in map(json.loads, src.open())]
    df = pd.DataFrame(rows, columns=["source_id", "text"])
    df["block_id"] = df["source_id"]
    return df


def _load_liar() -> pd.DataFrame:
    """LIAR, the TSV files of liar_dataset.zip, with all three splits.

    The loader reads only the statement (column 2) and the speaker (column 4).
    A block contains the statements of one speaker, because these statements
    are not independent.
    """
    src = RAW_X / "liar_dataset"
    tsvs = sorted(src.glob("*.tsv"))
    if not tsvs:
        sys.exit(f"missing {src}/*.tsv. Unzip liar_dataset.zip from "
                 "sites.cs.ucsb.edu/~william/")
    parts = [pd.read_csv(t, sep="\t", header=None, usecols=[2, 4],
                         dtype=str, quoting=3, on_bad_lines="error")
               .rename(columns={2: "text", 4: "block_id"})
             for t in tsvs]
    df = pd.concat(parts, ignore_index=True)
    df["source_id"] = ""
    df["block_id"] = df["block_id"].fillna("").replace("", "unknown_speaker")
    return df


# concept -> (prefix of the doc_id, loader)
CORPORA = {
    "hate_speech": ("hs", _load_civil_comments),
    "climate_contrarian": ("cc", _load_climate_fever),
    "fake_news": ("fn", _load_liar),
}

_WS = re.compile(r"\s+")


def _norm(t: str) -> str:
    return _WS.sub(" ", (t or "")).strip()


def draw(construct: str, prefix: str, loader) -> pd.DataFrame:
    """Draw the texts of one concept."""
    df = loader()
    df["text"] = df["text"].map(_norm)
    df["n_words"] = df["text"].str.split().str.len()
    pool = df[(df.n_words >= WORDS_MIN) & (df.n_words <= WORDS_MAX)]
    pool = pool.drop_duplicates(subset="text")
    # Each concept has its own seed, derived from SEED and its name.
    seed = SEED + int(hashlib.sha256(construct.encode()).hexdigest()[:6], 16) % 9973
    take = pool.sample(n=N_PER, random_state=seed)

    out = pd.DataFrame({
        "doc_id": [f"{prefix}:" + hashlib.sha256(t.encode()).hexdigest()[:12]
                   for t in take.text],
        "draw_cell": f"{construct}/expansion",
        "weight": 1.0,
        "construct": construct,
        "catalogue": "",
        "authored_by": "",
        "tier": "expansion",
        "lang": "en",
        "unit_text": take.text.values,
        "char_span": "",
        "article_id": take.source_id.astype(str).values,
        "pub_date": "",
        "n_words": take.n_words.values,
        "trigger_tags": "",
    })
    blocks = take.block_id.astype(str).values
    out["cluster_id"] = [b if b else d for b, d in zip(blocks, out.doc_id)]
    out["event_block_id"] = out["cluster_id"]
    dupes = out.doc_id.duplicated().sum()
    assert dupes == 0, f"{construct}: {dupes} doc_id collisions"
    return out


def main() -> None:
    frames = [draw(c, p, loader) for c, (p, loader) in CORPORA.items()]
    sample = pd.concat(frames, ignore_index=True)
    sha = frame_sha_of(sample)
    sample["frame_sha"] = sha

    DATA.mkdir(parents=True, exist_ok=True)
    dest = DATA / "sample_expansion.parquet"
    sample.to_parquet(dest, index=False)

    print(f"[done] {len(sample)} texts -> {dest}")
    print(f"[done] frame_sha {sha}")
    print(sample.groupby("construct").agg(
        n=("doc_id", "size"), blocks=("cluster_id", "nunique"),
        mean_words=("n_words", "mean")).round(1))


if __name__ == "__main__":
    main()
