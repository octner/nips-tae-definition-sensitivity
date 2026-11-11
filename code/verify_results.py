"""Check the results files and the analysis tables.

The code names a model "arm", a concept "construct", a prompt "instrument" and
definition sensitivity "transmission". The script recomputes values from the
inputs of an_transmission.py, separately from its NumPy matrices, and compares
them with the tables in these checks.

1. The manifest records seed 42, 2,000 bootstrap draws, the hash of the
   news-derived sample, 188 excluded texts and the rule for failed requests.
2. Each results file contains as many rows as the frozen samples have texts of
   its concept.
3. For each results file, n_missing in t1 equals the number of failed requests
   among the analysed texts.
4. For Gemma (arms[0]), these recomputed values agree with the tables within
   1e-9:
   - the INCLUDE share with the original wording of ACLED in t1
   - definition sensitivity and the paraphrase floor of protest in t2
   - the agreement between CCC 2021 and CCC 2024 in t3
   - the share of texts labelled INCLUDE with GTD with successful attacks only
     that are also labelled INCLUDE with GTD, in t5
5. For Gemma, each pair of wordings of a protest definition has at least 150
   analysed texts with two labels.

The script prints one PASS or FAIL line for each check, and its exit code is 0
when all checks pass.

Run from the repository root, after an_transmission.py.
    python3 code/verify_results.py
"""
import itertools
import json

import pandas as pd

from _paths import DATA, LABELS, TABLES, VERDICTS
from an_transmission import INSTRUMENTS

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" +
          (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def raw(arm, construct, def_id, impl):
    df = pd.read_json(LABELS / f"{arm}__{construct}__{def_id}__impl_{impl}.jsonl",
                      lines=True)
    df["verdict"] = df.verdict.where(df.verdict.isin(VERDICTS))
    return df.set_index("doc_id")


def main():
    sample = pd.concat([pd.read_parquet(DATA / f,
                                        columns=["doc_id", "construct", "lang"])
                        for f in ("sample.parquet", "sample_expansion.parquet")])
    # Terrorism uses the assassination texts.
    n_texts = sample.construct.value_counts()
    n_texts["terrorism"] = n_texts["assassination"]
    # The analysed texts, as in an_transmission.py.
    corrupt = set(pd.read_csv(DATA / "corrupt_fetch_ids.csv").doc_id)
    analysed = set(sample.doc_id[sample.lang.isin(("en", "es"))]) - corrupt

    man = json.loads((TABLES / "an_manifest.json").read_text())
    arms = man["arms"]

    print("[verify] manifest")
    check("seed 42", man["seed"] == 42)
    check("2000 draws", man["n_boot"] == 2000)
    check("frame_sha", man["frame_sha"] == "be53413d3dd1")
    check("corrupt 188", man["corrupt_excluded"] == 188)
    check("missing rule recorded", "pair dropped" in man["missing_rule"])

    t1 = pd.read_csv(TABLES / "t1_verdict_matrix.csv")
    t2 = pd.read_csv(TABLES / "t2_transmission_floors.csv")
    t3 = pd.read_csv(TABLES / "t3_ccc_contrast.csv")
    t5 = pd.read_csv(TABLES / "t5_nesting.csv")

    print("[verify] coverage and missing labels")
    for arm in arms:
        for construct in man["constructs"]:
            for d, impls in INSTRUMENTS[construct].items():
                for i in impls:
                    df = raw(arm, construct, d, i)
                    check(f"coverage {arm} {d}_{i}",
                          len(df) == n_texts[construct], f"{len(df)}")
                    keep = df[df.index.isin(analysed)]
                    t1row = t1[(t1.arm == arm) & (t1.construct == construct)
                               & (t1.instrument == f"{d}__impl_{i}")]
                    check(f"missingness {arm} {d}_{i}",
                          int(t1row.n_missing.iloc[0])
                          == int(keep.verdict.isna().sum()))

    print("[verify] spot reproduction")
    arm = arms[0]
    g = raw(arm, "protest", "acled", 0)
    g = g[g.index.isin(analysed)]
    share = (g.verdict == "INCLUDE").sum() / g.verdict.notna().sum()
    t1row = t1[(t1.arm == arm) & (t1.instrument == "acled__impl_0")]
    check("T1 acled INCLUDE share", abs(share - t1row.INCLUDE.iloc[0]) < 1e-9,
          f"{share:.6f}")

    # t2, definition sensitivity and the paraphrase floor
    wide = {}
    for d, impls in INSTRUMENTS["protest"].items():
        for i in impls:
            w = raw(arm, "protest", d, i)
            wide[f"{d}__impl_{i}"] = w[w.index.isin(analysed)].verdict
    wide = pd.DataFrame(wide)
    impl0 = [f"{d}__impl_0" for d in INSTRUMENTS["protest"]]
    cross = [(wide[a] != wide[b])[wide[a].notna() & wide[b].notna()].mean()
             for a, b in itertools.combinations(impl0, 2)]
    para_pairs = [p for d, impls in INSTRUMENTS["protest"].items()
                  for p in itertools.combinations(
                      [f"{d}__impl_{i}" for i in impls], 2)]
    para = [(wide[a] != wide[b])[wide[a].notna() & wide[b].notna()].mean()
            for a, b in para_pairs]
    t2row = t2[(t2.arm == arm) & (t2.construct == "protest")]
    check("T2 transmission", abs(sum(cross) / len(cross)
                                 - t2row.transmission.iloc[0]) < 1e-9)
    check("T2 floor", abs(sum(para) / len(para)
                          - t2row.paraphrase_floor.iloc[0]) < 1e-9)
    for a, b in para_pairs:
        n = (wide[a].notna() & wide[b].notna()).sum()
        check(f"floor pair n>=150 {a}~{b}", n >= 150, f"{n}")

    ok = wide["ccc_2021__impl_0"].notna() & wide["ccc_2024__impl_0"].notna()
    agr = (wide["ccc_2021__impl_0"] == wide["ccc_2024__impl_0"])[ok].mean()
    t3row = t3[t3.arm == arm]
    check("T3 agreement", abs(agr - t3row.agreement.iloc[0]) < 1e-9)

    ga = raw(arm, "assassination", "gtd", 0)
    gs = raw(arm, "assassination", "gtd_success", 0)
    j = ga.join(gs, lsuffix="_g", rsuffix="_s")
    j = j[j.index.isin(analysed)]
    m = j.verdict_g.notna() & (j.verdict_s == "INCLUDE")
    nest = (j.verdict_g[m] == "INCLUDE").mean()
    t5row = t5[t5.arm == arm]
    check("T5 nesting", abs(nest - t5row.nested_in_gtd.iloc[0]) < 1e-9)

    if FAILURES:
        raise SystemExit(f"[verify] {len(FAILURES)} FAILURES: {FAILURES}")
    print("[verify] ALL CHECKS PASS")


if __name__ == "__main__":
    main()
