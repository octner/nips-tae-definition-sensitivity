"""Compute tables t9, t10 and t11 on definition sensitivity.

The code names a model "arm", a concept "construct" and definition
sensitivity "transmission". The script uses the texts, the pairs of prompts
and the bootstrap of an_transmission.py, so its results are comparable with
t2_transmission_floors.csv.

1. For each model, the paraphrase floor of each paraphrased definition of
   protest, and the ratio of definition sensitivity to each paraphrase floor.
   Protest is the only concept with two paraphrased definitions, ACLED (acled)
   and MMPD (mm).
   -> TABLES/t9_floor_by_def.csv
2. For Sonnet on fake news, definition sensitivity, the paraphrase floor and
   the gap, with 95% confidence intervals. With each prompt, Sonnet labelled
   the texts in the same batches of at most 50. The bootstrap resamples these
   batches in place of the speakers of the texts.
   -> TABLES/t10_fake_batch_boot.csv
3. For each model and concept, the disagreement rate of each pair of competing
   definitions. The mean of these rates is definition sensitivity in
   t2_transmission_floors.csv.
   -> TABLES/t11_pairs.csv

The script reads the inputs of an_transmission.py,
TABLES/t2_transmission_floors.csv and the batch files in
SONNET_BATCHES/fake_news.

Run from the repository root, after an_transmission.py.
    python3 code/an_reviews.py
"""
import itertools
import json
import sys

import numpy as np
import pandas as pd

from _paths import CONCEPTS, MODELS, SONNET_BATCHES, TABLES
from an_transmission import (INSTRUMENTS, block_draws, ci, disagree_matrix,
                             load_arm, load_frame, mean_disagreement,
                             pair_sets)


def floor_by_definition(frame, t2):
    rows = []
    for arm in MODELS:
        wide = load_arm(arm, "protest", frame)
        trans = t2[(t2.arm == arm) & (t2.construct == "protest")].iloc[0].transmission
        for d, impls in INSTRUMENTS["protest"].items():
            if len(impls) < 2:
                continue
            pairs = list(itertools.combinations(
                [f"{d}__impl_{i}" for i in impls], 2))
            fl = mean_disagreement(disagree_matrix(wide, pairs))
            rows.append({"arm": arm, "def_id": d, "floor": fl,
                         "transmission": trans, "ratio": trans / fl})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "t9_floor_by_def.csv", index=False)
    print(df.to_string(index=False))


def fake_batch_boot(frame):
    arm, con = "claude_sonnet5", "fake_news"
    wide = load_arm(arm, con, frame)
    # For each prompt, the last part of each file name is the batch of its
    # texts.
    pay = SONNET_BATCHES / con
    maps = {}
    for inst in sorted(p.name for p in pay.iterdir() if p.is_dir()):
        m = {}
        for f in sorted((pay / inst).glob("*.json")):
            c = f.stem.split("__")[-1]
            for u in json.loads(f.read_text())["units"]:
                m[u["doc_id"]] = c
        maps[inst] = m
    first = next(iter(maps.values()))
    if any(m != first for m in maps.values()):
        sys.exit("The fake_news batches differ between prompts.")
    batch = pd.Series(first).reindex(wide.index)
    if batch.isna().any():
        sys.exit("Some fake_news texts have no batch.")
    cross, para = pair_sets(con)
    mc, mp = disagree_matrix(wide, cross), disagree_matrix(wide, para)
    bt = np.array([[mean_disagreement(mc, r), mean_disagreement(mp, r)]
                   for r in block_draws(batch.to_numpy())])
    trans, floor = mean_disagreement(mc), mean_disagreement(mp)
    trans_lo, trans_hi = ci(bt[:, 0])
    floor_lo, floor_hi = ci(bt[:, 1])
    gap_lo, gap_hi = ci(bt[:, 0] - bt[:, 1])
    row = {"arm": arm, "construct": con, "n_batches": int(batch.nunique()),
           "transmission": trans, "transmission_lo": trans_lo,
           "transmission_hi": trans_hi, "paraphrase_floor": floor,
           "floor_lo": floor_lo, "floor_hi": floor_hi,
           "gap_points": trans - floor, "gap_lo": gap_lo, "gap_hi": gap_hi}
    pd.DataFrame([row]).to_csv(TABLES / "t10_fake_batch_boot.csv", index=False)
    print(row)


def pair_rates(frame):
    rows = []
    for con in CONCEPTS:
        cross, _ = pair_sets(con)
        for arm in MODELS:
            wide = load_arm(arm, con, frame)
            rate = np.nanmean(disagree_matrix(wide, cross), axis=0)
            for (a, b), r in zip(cross, rate):
                rows.append({"arm": arm, "construct": con,
                             "def_a": a.split("__")[0],
                             "def_b": b.split("__")[0], "rate": float(r)})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "t11_pairs.csv", index=False)
    print(df.groupby(["construct", "arm"]).rate.mean())


def main():
    frame = load_frame()
    t2 = pd.read_csv(TABLES / "t2_transmission_floors.csv")
    floor_by_definition(frame, t2)
    fake_batch_boot(frame)
    pair_rates(frame)


if __name__ == "__main__":
    main()
