"""Compute tables t7 and t8 for Sonnet's repeat batches and reasoning effort.

The open-weight models ran at temperature 0, but Sonnet ran without
temperature control, so the paper measures its repeat floor. The repeat floor
is the share of labels that change when the same prompt runs again on the same
texts. The code names a model "arm", a concept "construct" and a batch
"chunk".

For each repeat batch, t7_repeat.csv contains the share of labels that are the
same as in the main annotation run. The script compares only the analysed
texts of an_transmission.py with a label in both runs.

For each pair of the three wordings of the ACLED definition, t8_effort.csv
contains the disagreement rate. With the original wording (impl_0) only, we
changed Sonnet's reasoning effort partway through the texts. Column
straddles_split is True for the two pairs that contain impl_0.

The script reads the frozen samples, DATA/corrupt_fetch_ids.csv, Sonnet's
results files in LABELS and SONNET_REPEAT/*__rpt.json.result.json. It writes
TABLES/t7_repeat.csv and TABLES/t8_effort.csv.

Run from the repository root.
    python3 code/an_repeat.py
"""
import json

import pandas as pd

from _paths import LABELS, LATER_CONCEPTS, SONNET_REPEAT, TABLES, VERDICTS
from an_transmission import load_frame

ARM = "claude_sonnet5"
EFFORT_MIXED = "protest__acled__impl_0"
EFFORT_UNIFORM = ("protest__acled__impl_1", "protest__acled__impl_2")


def labels(name, analysed):
    """doc_id -> Sonnet's label for one prompt, or None for a failed request."""
    v = {}
    for line in (LABELS / f"{ARM}__{name}.jsonl").open():
        r = json.loads(line)
        if r["doc_id"] in analysed:
            v[r["doc_id"]] = r["verdict"] if r["verdict"] in VERDICTS else None
    return v


def repeat_floor(analysed):
    # Rows for protest and assassination come first, then the other four
    # concepts.
    files = sorted(SONNET_REPEAT.glob("*__rpt.json.result.json"),
                   key=lambda p: (p.name.split("__")[0] in LATER_CONCEPTS,
                                  p.name))
    rows = []
    for p in files:
        # protest__acled__impl_0__c000__rpt.json.result.json
        stem = p.name.split("__rpt")[0]
        construct, def_id, impl, chunk = stem.split("__")
        main_run = labels(f"{construct}__{def_id}__{impl}", analysed)
        n = agree = 0
        for r in json.loads(p.read_text()):
            if r["doc_id"] not in analysed:
                continue
            a = main_run.get(r["doc_id"])
            b = r["verdict"] if r["verdict"] in VERDICTS else None
            if a is None or b is None:
                continue
            n += 1
            agree += (a == b)
        rows.append({"arm": ARM, "construct": construct, "def_id": def_id,
                     "impl_id": impl, "chunk": chunk,
                     "n_compared": n, "n_identical": agree,
                     "agreement": agree / n})
        print(f"  {construct}/{def_id}/{impl} {chunk}: {agree}/{n} "
              f"({100*agree/n:.1f}%)")
    out = TABLES / "t7_repeat.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[repeat] wrote {out}")


def effort_contrast(analysed):
    def dis(a, b):
        m = pd.DataFrame({"a": pd.Series(labels(a, analysed)),
                          "b": pd.Series(labels(b, analysed))}).dropna()
        return float((m.a != m.b).mean()), len(m)

    rows = []
    for other in EFFORT_UNIFORM:
        d, n = dis(EFFORT_MIXED, other)
        rows.append({"pair": f"{EFFORT_MIXED}~{other}", "straddles_split": True,
                     "disagreement": d, "n": n})
    a, b = EFFORT_UNIFORM
    d, n = dis(a, b)
    rows.append({"pair": f"{a}~{b}", "straddles_split": False,
                 "disagreement": d, "n": n})
    df = pd.DataFrame(rows)
    out = TABLES / "t8_effort.csv"
    df.to_csv(out, index=False)
    mixed = df[df.straddles_split].disagreement.mean()
    clean = df[~df.straddles_split].disagreement.mean()
    print(f"[effort] pairs with impl_0 {100*mixed:.2f}%, other pair "
          f"{100*clean:.2f}%, difference {100*(mixed-clean):+.2f} points")
    print(f"[effort] wrote {out}")


def main():
    analysed = set(load_frame().index)
    repeat_floor(analysed)
    effort_contrast(analysed)


if __name__ == "__main__":
    main()
