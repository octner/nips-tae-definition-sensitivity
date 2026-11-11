"""Compare the labels of the main annotation run and of the re-run on a second
platform.

Both open-weight models labelled the texts of terrorism, hate speech, climate
contrarianism and fake news again on a second platform. For each prompt, the
script counts the texts with a label in both runs and the texts with the same
label in both runs. The code names a model "arm", a concept "construct" and a
prompt "instrument".

The script reads the results files in LABELS and in LABELS_SECOND_PLATFORM.
It writes TABLES/t6_crosssite.csv, with one row for each model, concept and
prompt. Columns n_fail_banked and n_fail_other contain the number of failed
requests in the main annotation run and in the re-run. Column n_fail_disjoint
contains the number of texts with a failed request in only one of the two
runs.

Run from the repository root.
    python3 code/an_crosssite.py
"""
import json

import pandas as pd

from _paths import LABELS, LABELS_SECOND_PLATFORM, TABLES, VERDICTS


def load(path):
    """doc_id -> label, or None for a failed request."""
    out = {}
    for line in path.open():
        d = json.loads(line)
        out[d["doc_id"]] = d["verdict"] if d["verdict"] in VERDICTS else None
    return out


def main():
    rows = []
    for m in sorted(LABELS_SECOND_PLATFORM.glob("*.jsonl")):
        a, c = load(LABELS / m.name), load(m)
        keys = set(a) & set(c)
        both = [k for k in keys if a[k] is not None and c[k] is not None]
        arm, construct, def_id, impl = m.stem.split("__")
        fa = {k for k in keys if a[k] is None}
        fc = {k for k in keys if c[k] is None}
        rows.append({
            "arm": arm, "construct": construct,
            "instrument": f"{def_id}__{impl}",
            "n_compared": len(both),
            "n_identical": sum(1 for k in both if a[k] == c[k]),
            "n_fail_banked": len(fa), "n_fail_other": len(fc),
            "n_fail_disjoint": len(fa ^ fc),
        })

    t6 = pd.DataFrame(rows)
    t6["agreement"] = t6.n_identical / t6.n_compared
    out = TABLES / "t6_crosssite.csv"
    t6.to_csv(out, index=False)

    print("[crosssite] main run against the second platform")
    for arm, g in t6.groupby("arm"):
        n, k = int(g.n_compared.sum()), int(g.n_identical.sum())
        print(f"  {arm:14} {k:,}/{n:,} identical = {100 * k / n:.4f}%  "
              f"| texts that failed in one run only: "
              f"{int(g.n_fail_disjoint.sum())}"
              f"  | lowest prompt {g.agreement.min() * 100:.2f}%")
    print(f"[crosssite] wrote {out}")


if __name__ == "__main__":
    main()
