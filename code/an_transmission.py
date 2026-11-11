"""Compute definition sensitivity and the paraphrase floor of each model and
concept.

The code names a model "arm", a concept "construct", a prompt "instrument"
and definition sensitivity "transmission". A prompt pairs one wording of a
definition with a query. The IDs impl_0, impl_1 and impl_2 mark the original
wording and its paraphrases.

The disagreement rate of two prompts is the share of texts with different
labels. Definition sensitivity is the mean disagreement rate over the pairs of
competing definitions in their original wording. The paraphrase floor is the
mean disagreement rate over the pairs of wordings of one definition. The gap
is definition sensitivity minus the paraphrase floor, and the ratio is their
quotient.

The analysed texts are the English and Spanish texts of the frozen samples,
without the 188 assassination texts that the download returned as binary
files. The script drops each pair of labels with a missing label from a
failed request.

A block bootstrap with 2,000 draws and seed 42 produces the 95% confidence
intervals, and column event_block_id contains the block IDs.

The script reads DATA/sample.parquet, DATA/sample_expansion.parquet,
DATA/corrupt_fetch_ids.csv and the results files in LABELS. It writes these
files in TABLES:
  - t1_verdict_matrix.csv       the share of each label for each prompt
  - t2_transmission_floors.csv  definition sensitivity, the paraphrase floor,
                                the gap and the ratio
  - t3_ccc_contrast.csv         the agreement between CCC 2021 and CCC 2024,
                                and the change in the INCLUDE share
  - t3_ccc_flips__<model>.csv   the number of texts for each pair of labels
                                with CCC 2021 and CCC 2024
  - t5_nesting.csv              the number of texts labelled INCLUDE with GTD
                                with successful attacks only, and the share of
                                them also labelled INCLUDE with GTD
  - an_manifest.json            the settings, the sample hashes and the number
                                of analysed texts

Run from the repository root.
    python3 code/an_transmission.py
"""
import itertools
import json

import numpy as np
import pandas as pd

from _paths import CONCEPTS, DATA, LABELS, MODELS, TABLES, VERDICTS

N_BOOT = 2000
SEED = 42

# The definitions of each concept and their wordings.
INSTRUMENTS = {
    "protest": {"acled": [0, 1, 2], "ccc_2021": [0], "ccc_2024": [0],
                "cnts_agd": [0], "mm": [0, 1, 2]},
    "assassination": {"cnts": [0], "gtd": [0, 1, 2], "gtd_success": [0],
                      "targeted_killing": [0]},
    "hate_speech": {"davidson": [0, 1, 2], "degibert": [0], "founta": [0],
                    "waseem_hovy": [0]},
    "climate_contrarian": {"capstick": [0], "cards": [0, 1, 2], "cook": [0],
                           "rahmstorf": [0]},
    "fake_news": {"allcott": [0], "gelfert": [0], "lazer": [0, 1, 2],
                  "zhou": [0]},
    "terrorism": {"enders_sandler": [0], "gtd_crit": [0, 1, 2],
                  "schmid": [0], "weinberg": [0]},
}


def corrupt_ids():
    """The doc_ids of the excluded assassination texts."""
    return set(pd.read_csv(DATA / "corrupt_fetch_ids.csv").doc_id)


def load_frame():
    """The analysed texts of all six concepts."""
    cols = ["doc_id", "construct", "event_block_id", "lang", "frame_sha"]
    s = pd.concat([pd.read_parquet(DATA / "sample.parquet", columns=cols),
                   pd.read_parquet(DATA / "sample_expansion.parquet",
                                   columns=cols)])
    s = s[~s.doc_id.isin(corrupt_ids()) & s.lang.isin(("en", "es"))]
    # Terrorism uses the assassination texts.
    t = s[s.construct == "assassination"].assign(construct="terrorism")
    return pd.concat([s, t]).set_index("doc_id")


def load_arm(arm, construct, frame):
    """The labels of one model on one concept.

    One row for each analysed text and one column for each prompt
    'def__impl_N'. A missing label is NaN.
    """
    idx = frame.index[frame.construct == construct]
    cols = {}
    for d, impls in INSTRUMENTS[construct].items():
        for i in impls:
            v = {}
            f = LABELS / f"{arm}__{construct}__{d}__impl_{i}.jsonl"
            for line in f.open():
                r = json.loads(line)
                v[r["doc_id"]] = (r["verdict"]
                                  if r["verdict"] in VERDICTS else np.nan)
            cols[f"{d}__impl_{i}"] = pd.Series(v)
    return pd.DataFrame(cols).reindex(idx)


def pair_sets(construct):
    """The pairs of prompts for definition sensitivity and the paraphrase
    floor."""
    defs = INSTRUMENTS[construct]
    cross = list(itertools.combinations([f"{d}__impl_0" for d in defs], 2))
    para = [p for d, impls in defs.items()
            for p in itertools.combinations(
                [f"{d}__impl_{i}" for i in impls], 2)]
    return cross, para


def disagree_matrix(wide, pairs):
    """Texts x pairs. 1 is disagreement, 0 is agreement and NaN is a missing
    label."""
    out = np.full((len(wide), len(pairs)), np.nan)
    for j, (a, b) in enumerate(pairs):
        va, vb = wide[a].to_numpy(object), wide[b].to_numpy(object)
        ok = pd.notna(va) & pd.notna(vb)
        out[ok, j] = (va[ok] != vb[ok]).astype(float)
    return out


def block_draws(blocks):
    """The row numbers of each bootstrap draw."""
    rng = np.random.default_rng(SEED)
    uniq = pd.unique(blocks)
    rows_of = pd.Series(blocks).groupby(blocks).indices
    draws = []
    for _ in range(N_BOOT):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        draws.append(np.concatenate([rows_of[b] for b in picked]))
    return draws


def mean_disagreement(mat, rows=None):
    """The mean disagreement rate over the pairs."""
    m = mat if rows is None else mat[rows]
    per_pair = np.nanmean(m, axis=0)
    return float(np.nanmean(per_pair))


def ci(vals):
    """The 95% percentile interval."""
    return tuple(np.percentile(vals, [2.5, 97.5]))


def analyze(arm, construct, frame, draws):
    wide = load_arm(arm, construct, frame)
    cross, para = pair_sets(construct)
    mat_c = disagree_matrix(wide, cross)
    mat_p = disagree_matrix(wide, para)

    # t2
    trans = mean_disagreement(mat_c)
    floor = mean_disagreement(mat_p)
    bt = np.array([[mean_disagreement(mat_c, r),
                    mean_disagreement(mat_p, r)] for r in draws])
    trans_lo, trans_hi = ci(bt[:, 0])
    floor_lo, floor_hi = ci(bt[:, 1])
    gap_lo, gap_hi = ci(bt[:, 0] - bt[:, 1])
    t2 = {"arm": arm, "construct": construct,
          "transmission": trans, "transmission_lo": trans_lo,
          "transmission_hi": trans_hi,
          "paraphrase_floor": floor, "floor_lo": floor_lo,
          "floor_hi": floor_hi,
          "gap_points": trans - floor, "gap_lo": gap_lo, "gap_hi": gap_hi,
          "ratio": trans / floor,
          "n_cross_pairs": len(cross), "n_para_pairs": len(para)}

    # t1
    t1 = []
    for col in wide.columns:
        v = wide[col]
        ok = v.notna().to_numpy()
        row = {"arm": arm, "construct": construct, "instrument": col,
               "n_missing": int((~ok).sum())}
        for lev in VERDICTS:
            m = (v == lev).to_numpy()
            row[lev] = m[ok].mean()
        t1.append(row)

    # t3, protest only
    t3 = None
    if construct == "protest":
        a, b = wide["ccc_2021__impl_0"], wide["ccc_2024__impl_0"]
        ok = a.notna() & b.notna()
        mat_ccc = disagree_matrix(wide, [("ccc_2021__impl_0",
                                          "ccc_2024__impl_0")])
        dis_lo, dis_hi = ci([mean_disagreement(mat_ccc, r) for r in draws])
        inc_d = ((b == "INCLUDE").astype(float) -
                 (a == "INCLUDE").astype(float)).to_numpy()
        inc_d[~ok.to_numpy()] = np.nan
        inc_lo, inc_hi = ci([np.nanmean(inc_d[r]) for r in draws])
        t3 = {"arm": arm, "n": int(ok.sum()),
              "agreement": 1 - mean_disagreement(mat_ccc),
              "agreement_lo": 1 - dis_hi, "agreement_hi": 1 - dis_lo,
              "include_delta": float(np.nanmean(inc_d)),
              "include_delta_lo": inc_lo, "include_delta_hi": inc_hi}
        flips = pd.crosstab(a[ok], b[ok], rownames=["ccc_2021"],
                            colnames=["ccc_2024"])
        flips.to_csv(TABLES / f"t3_ccc_flips__{arm}.csv")

    # t5, assassination only. A text labelled INCLUDE with GTD with successful
    # attacks only should also be labelled INCLUDE with GTD, which has one
    # requirement fewer.
    t5 = None
    if construct == "assassination":
        g, gs = wide["gtd__impl_0"], wide["gtd_success__impl_0"]
        ok = g.notna() & gs.notna()
        inc = gs[ok] == "INCLUDE"
        t5 = {"arm": arm, "n_gtd_success_include": int(inc.sum()),
              "nested_in_gtd": float((g[ok][inc] == "INCLUDE").mean())}
    return t1, t2, t3, t5


def main():
    frame = load_frame()
    # All models share the draws, which depend only on the blocks.
    draws = {c: block_draws(frame.event_block_id[frame.construct == c].to_numpy())
             for c in CONCEPTS}
    t1s, t2s, t3s, t5s = [], [], [], []
    for arm in MODELS:
        for construct in CONCEPTS:
            t1, t2, t3, t5 = analyze(arm, construct, frame, draws[construct])
            t1s += t1
            t2s.append(t2)
            if t3:
                t3s.append(t3)
            if t5:
                t5s.append(t5)
            print(f"[{arm}/{construct}] transmission {t2['transmission']:.3f} "
                  f"[{t2['transmission_lo']:.3f},{t2['transmission_hi']:.3f}] "
                  f"floor {t2['paraphrase_floor']:.3f} "
                  f"[{t2['floor_lo']:.3f},{t2['floor_hi']:.3f}] "
                  f"ratio {t2['ratio']:.1f}x")
    pd.DataFrame(t1s).to_csv(TABLES / "t1_verdict_matrix.csv", index=False)
    pd.DataFrame(t2s).to_csv(TABLES / "t2_transmission_floors.csv", index=False)
    pd.DataFrame(t3s).to_csv(TABLES / "t3_ccc_contrast.csv", index=False)
    pd.DataFrame(t5s).to_csv(TABLES / "t5_nesting.csv", index=False)
    # The hash of each frozen sample.
    sha = frame.groupby("construct").frame_sha.first()
    (TABLES / "an_manifest.json").write_text(json.dumps({
        "arms": list(MODELS), "n_boot": N_BOOT, "seed": SEED,
        "frame_sha": sha["protest"], "corrupt_excluded": len(corrupt_ids()),
        "constructs": list(CONCEPTS),
        "frame_sha_expansion": sha["hate_speech"],
        "analyzable": {c: int((frame.construct == c).sum())
                       for c in CONCEPTS},
        "missing_rule": "API_FAIL verdict=null -> pair dropped",
    }, indent=1))
    print(f"[an] tables written to {TABLES}")


if __name__ == "__main__":
    main()
