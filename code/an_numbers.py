r"""Write the numbers and three tables of the paper as LaTeX files in paper/.

The paper contains these numbers as LaTeX macros, for example
\taeAllTransMin.

The code names a model "arm", a concept "construct", a prompt "instrument"
and a text "unit". It names a label "verdict", definition sensitivity
"transmission" and the comprehension check "G2". The ID impl_0 marks the
original wording of a definition, and impl_1 and impl_2 mark its paraphrases.

The script reads the tables in tables/, the prompt registry and the answers
in the comprehension check. It also reads Sonnet's labels and batches for
fake news. For the other four concepts, it reads Qwen's labels and the files
of Qwen's first run. It runs verify_results.py and counts the checks that
pass.

It writes the macros to numbers.tex and the bodies of three tables of
Section 4 to these files:

    - tab_all.tex     definition sensitivity and the paraphrase floor
    - tab_pairs.tex   the closest and the farthest pair of definitions
    - tab_shares.tex  the range of the INCLUDE and INDETERMINATE shares
                      across definitions

Run from the repository root.

    python3 code/an_numbers.py
"""
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from _paths import (COMPREHENSION, LABELS, LABELS_QWEN_FIRST_RUN,
                    LATER_CONCEPTS, MODELS, PAPER, REGISTRY, SONNET_BATCHES,
                    TABLES)
from gate import G2_KEY

SUFFIX = {"gemma4_31b": "Gemma", "qwen36_27b": "Qwen",
          "claude_sonnet5": "Sonnet"}
MODEL_NAME = {"gemma4_31b": "Gemma 4 31B", "qwen36_27b": "Qwen 3.6 27B",
              "claude_sonnet5": "Claude Sonnet 5"}
# The concepts in the order of the tables of the paper.
ORDER = ["protest", "assassination", "terrorism", "hate_speech",
         "climate_contrarian", "fake_news"]
CONCEPT_NAME = {"protest": "Protest", "assassination": "Assassination",
                "terrorism": "Terrorism", "hate_speech": "Hate speech",
                "climate_contrarian": "Climate contrarianism",
                "fake_news": "Fake news"}
# The part of a macro name for each protest definition.
PROTEST_MACRO = {"acled": "Acled", "ccc_2021": "CccOld", "ccc_2024": "CccNew",
                 "cnts_agd": "Cnts", "mm": "Mm"}
# The names of the definitions in the table of pairs.
DEF_NAME = {"protest": {"acled": "ACLED", "ccc_2021": "CCC 2021",
                        "ccc_2024": "CCC 2024",
                        "cnts_agd": "Cross-National Time-Series Data Archive (CNTS)",
                        "mm": "MMPD"},
            "assassination": {"gtd": "GTD", "cnts": "CNTS",
                              "gtd_success": "GTD with successful attacks only",
                              "targeted_killing": "\\citeauthor{icrc2026targeted}"},
            "terrorism": {"gtd_crit": "GTD",
                          "schmid": "\\citeauthor{schmid2011revised}",
                          "enders_sandler": "\\citeauthor{enders2011terrorism}",
                          "weinberg": "\\citeauthor{weinberg2004challenges}"},
            "hate_speech": {"davidson": "\\citeauthor{davidson2017automated}",
                            "waseem_hovy": "\\citeauthor{waseem2016hateful}",
                            "founta": "\\citeauthor{founta2018large}",
                            "degibert": "\\citeauthor{degibert2018hate}"},
            "climate_contrarian": {"cards": "\\citeauthor{coan2021computer}",
                                   "cook": "\\citeauthor{cook2013quantifying}",
                                   "capstick": "\\citeauthor{capstick2014what}",
                                   "rahmstorf": "\\citeauthor{rahmstorf2004climate}"},
            "fake_news": {"lazer": "\\citeauthor{lazer2018science}",
                          "allcott": "\\citeauthor{allcott2017social}",
                          "gelfert": "\\citeauthor{gelfert2018fake}",
                          "zhou": "\\citeauthor{zhou2020survey}"}}
WORDS = ("zero one two three four five six seven eight nine ten eleven "
         "twelve").split()


def stop(msg):
    raise SystemExit(f"an_numbers: {msg}")


def pct(x):
    """A fraction as a percentage with one decimal, for example 38.4 or -5.6."""
    return f"{100 * x:.1f}"


def thou(n):
    """A count with commas between the thousands, for example 3,249."""
    return f"{int(n):,}"


def verdicts(path):
    """The label of each text in one results file."""
    rows = map(json.loads, path.read_text().splitlines())
    return {r["doc_id"]: r["verdict"] for r in rows}


def comprehension_score(g, definitions):
    """The number of questions and the number of accepted answers.

    The table g contains the answers of one model in the comprehension check.
    Only the definitions in the registry count. An answer is accepted if it is
    in the set of acceptable answers in gate.py.
    """
    g = g[[(c, d) in definitions for c, d in zip(g.construct, g.def_id)]]
    right = sum(r.answer in G2_KEY[(r.construct, r.def_id)][r.question]
                for r in g.itertuples())
    return len(g), right


def main():
    t1 = pd.read_csv(TABLES / "t1_verdict_matrix.csv")
    t2 = pd.read_csv(TABLES / "t2_transmission_floors.csv")
    t3 = pd.read_csv(TABLES / "t3_ccc_contrast.csv")
    manifest = json.loads((TABLES / "an_manifest.json").read_text())
    n_texts = manifest["analyzable"]
    registry = json.loads(REGISTRY.read_text())
    definitions = {(r["construct"], r["def_id"]) for r in registry}
    # The share of each label with the original wording of each definition.
    orig = t1[t1.instrument.str.endswith("__impl_0")].copy()
    orig["def_id"] = orig.instrument.str.replace("__impl_0", "", regex=False)

    def row(model, concept):
        """The row of t2 for one model and one concept."""
        return t2[(t2.arm == model) & (t2.construct == concept)].iloc[0]

    lines = []

    def cmd(name, value):
        lines.append(f"\\newcommand{{\\tae{name}}}{{{value}}}")

    # In the paper, some numbers are inside these LaTeX switches. The script
    # writes all these numbers, so it sets each switch to true.
    for s in ("sonnet", "sonnetbreadth", "crosssite", "repeat", "rerun"):
        lines += [f"\\newif\\if{s}", f"\\{s}true"]

    # The evaluation setup -------------------------------------------------
    # The news-derived sample contains the analysed texts of protest and
    # assassination and the assassination texts that we excluded.
    cmd("NUnitsSample", thou(n_texts["protest"] + n_texts["assassination"]
                             + manifest["corrupt_excluded"]))
    cmd("NHate", thou(n_texts["hate_speech"]))
    cmd("NAnalyzableProtest", thou(n_texts["protest"]))
    cmd("NCorrupt", manifest["corrupt_excluded"])
    cmd("NInstr", len(registry))
    cmd("NDefs", len(definitions))
    n_defs = Counter(c for c, _ in definitions)
    cmd("NDefsProtestWord", WORDS[n_defs["protest"]])
    other = {n for c, n in n_defs.items() if c != "protest"}
    if len(other) != 1:
        stop("the other concepts have different numbers of definitions")
    cmd("NDefsOtherWord", WORDS[other.pop()])
    cmd("NCrossPairs", int(row(MODELS[0], "protest").n_cross_pairs))
    other_pairs = {int(n) for n in t2[t2.construct != "protest"].n_cross_pairs}
    if len(other_pairs) != 1:
        stop("the other concepts have different numbers of definition pairs")
    cmd("NCrossPairsOther", other_pairs.pop())
    cmd("NBoot", thou(manifest["n_boot"]))
    cmd("NPilot", 300)  # The pilot study used 300 protest texts.
    cmd("AllConstructWord", WORDS[t2.construct.nunique()])
    cmd("ArmCountWord", WORDS[len(MODELS)])
    names = [MODEL_NAME[m] for m in MODELS]
    cmd("ArmPhrase", f"{WORDS[len(MODELS)]} models "
                     f"({', '.join(names[:-1])}, and {names[-1]})")

    # Each model that took the comprehension check on a set of concepts must
    # answer the same number of questions.
    answers = {m: pd.read_csv(COMPREHENSION / f"g2_{m}.csv") for m in MODELS}

    def comprehension(prefix, concepts):
        scores = {}
        for m, g in answers.items():
            n, right = comprehension_score(g[g.construct.isin(concepts)],
                                           definitions)
            if n:
                scores[m] = (n, right)
        n_questions = {n for n, _ in scores.values()}
        if len(n_questions) != 1:
            stop(f"{prefix}: the models answered different numbers of questions")
        cmd(f"{prefix}Den", n_questions.pop())
        for m, (_, right) in scores.items():
            cmd(f"{prefix}{SUFFIX[m]}", right)

    comprehension("GTwo", {"protest", "assassination"})
    # Only Gemma and Qwen took the check on the other four concepts.
    comprehension("GTwoBreadth", LATER_CONCEPTS)

    # Definition sensitivity, the paraphrase floor and the label shares ----
    if (t2.paraphrase_floor >= t2.transmission).any():
        stop("a paraphrase floor is not below its definition sensitivity")
    for q, col in (("Trans", "transmission"), ("Floor", "paraphrase_floor")):
        cmd(f"All{q}Min", pct(t2[col].min()))
        cmd(f"All{q}Max", pct(t2[col].max()))
    cmd("AllRatioMin", f"{t2.ratio.min():.1f}")
    cmd("AllRatioMax", f"{t2.ratio.max():.1f}")

    # The paraphrase floor of each of the two paraphrased protest definitions.
    t9 = pd.read_csv(TABLES / "t9_floor_by_def.csv")
    for r in t9.itertuples():
        cmd(f"Floor{r.def_id.capitalize()}{SUFFIX[r.arm]}", pct(r.floor))
    cmd("FloorByDefRatioMin", f"{t9.ratio.min():.1f}")
    cmd("FloorByDefRatioMax", f"{t9.ratio.max():.1f}")

    # The disagreement rate of each pair of definitions. The mean of the rates
    # is definition sensitivity.
    t11 = pd.read_csv(TABLES / "t11_pairs.csv")
    for (m, c), g in t11.groupby(["arm", "construct"]):
        if abs(g.rate.mean() - row(m, c).transmission) > 1e-9:
            stop(f"t11 {m} {c}: the mean over pairs is not definition sensitivity")
    # For each concept, the closest pair, the farthest pair and the rate of
    # each pair for each model.
    pairs = {}
    for c in ORDER:
        w = t11[t11.construct == c].pivot_table(
            index=["def_a", "def_b"], columns="arm", values="rate")[list(MODELS)]
        mean = w.mean(axis=1)
        pairs[c] = (mean.idxmin(), mean.idxmax(), w)
    close, far, w = pairs["assassination"]
    if (close, far) != (("gtd", "gtd_success"), ("gtd", "targeted_killing")):
        stop("the closest or the farthest pair for assassination is not the "
             "pair that Section 4 names")
    cmd("PairAssnCloseLo", pct(w.loc[close].min()))
    cmd("PairAssnCloseHi", pct(w.loc[close].max()))
    cmd("PairAssnFarLo", pct(w.loc[far].min()))
    cmd("PairAssnFarHi", pct(w.loc[far].max()))

    # Sonnet on fake news.
    r = row("claude_sonnet5", "fake_news")
    cmd("TransFakeSonnet", pct(r.transmission))
    cmd("FloorFakeSonnet", pct(r.paraphrase_floor))
    # With most fake news prompts, Sonnet assigns one label to at least 96%
    # of the texts. Count these prompts.
    shares = []
    for f in sorted(LABELS.glob("claude_sonnet5__fake_news__*.jsonl")):
        counts = Counter(verdicts(f).values())
        shares.append(max(counts.values()) / sum(counts.values()))
    one_label = [x for x in shares if x >= 0.96]
    cmd("SonnetFakeInstrWord", WORDS[len(shares)])
    cmd("SonnetFakeDegenWord", WORDS[len(one_label)])
    cmd("SonnetFakeDegenShareMin", pct(min(one_label)))
    # On some whole batches, Sonnet's label with the original wording of
    # Lazer et al. is not its label with the paraphrases. Count these batches.
    batch = {}
    for f in (SONNET_BATCHES / "fake_news" / "lazer__impl_0").glob("*.json"):
        for u in json.loads(f.read_text())["units"]:
            batch[u["doc_id"]] = f.stem.split("__")[-1]
    lazer = [verdicts(LABELS / f"claude_sonnet5__fake_news__lazer__impl_{i}.jsonl")
             for i in range(3)]
    if lazer[1] != lazer[2]:
        stop("the two paraphrases of Lazer et al. give different labels")
    split = {batch[d] for d, v in lazer[0].items() if v != lazer[1].get(d)}
    if any(v == lazer[1].get(d) for d, v in lazer[0].items() if batch[d] in split):
        stop("in a batch of Lazer et al., only some labels change")
    cmd("SonnetFakeSplitBatchWord", WORDS[len(split)])
    # The bootstrap that resamples Sonnet's batches of fake news.
    fb = pd.read_csv(TABLES / "t10_fake_batch_boot.csv").iloc[0]
    cmd("FakeBatchWord", WORDS[int(fb.n_batches)])
    for q, col in (("Trans", "transmission"), ("Floor", "floor"), ("Gap", "gap")):
        cmd(f"FakeBatch{q}Lo", pct(fb[f"{col}_lo"]))
        cmd(f"FakeBatch{q}Hi", pct(fb[f"{col}_hi"]))

    # The share of the texts labelled INCLUDE with GTD with successful attacks
    # only that are also labelled INCLUDE with GTD.
    t5 = pd.read_csv(TABLES / "t5_nesting.csv")
    cmd("NestLo", pct(t5.nested_in_gtd.min()))
    cmd("NestHi", pct(t5.nested_in_gtd.max()))

    # The label shares for protest.
    protest = orig[orig.construct == "protest"]
    for m in MODELS:
        sub = protest[protest.arm == m].set_index("def_id")
        for d, part in PROTEST_MACRO.items():
            cmd(f"Inc{part}{SUFFIX[m]}", pct(sub.loc[d, "INCLUDE"]))
        cmd(f"IndetMin{SUFFIX[m]}", pct(sub.INDETERMINATE.min()))
        cmd(f"IndetMax{SUFFIX[m]}", pct(sub.INDETERMINATE.max()))
    cmd("IncSpanLo", pct(protest.INCLUDE.min()))
    cmd("IncSpanHi", pct(protest.INCLUDE.max()))
    cmd("IndetSpanLo", pct(protest.INDETERMINATE.min()))
    cmd("IndetSpanHi", pct(protest.INDETERMINATE.max()))

    # CCC 2021 against CCC 2024.
    for m in MODELS:
        r = t3[t3.arm == m].iloc[0]
        cmd(f"CccAgr{SUFFIX[m]}", pct(r.agreement))
        cmd(f"CccDelta{SUFFIX[m]}", pct(r.include_delta))
        cmd(f"CccDeltaLo{SUFFIX[m]}", pct(r.include_delta_lo))
        cmd(f"CccDeltaHi{SUFFIX[m]}", pct(r.include_delta_hi))
        flips = pd.read_csv(TABLES / f"t3_ccc_flips__{m}.csv", index_col=0)
        cmd(f"CccFlipStrict{SUFFIX[m]}", int(flips.loc["INCLUDE", "EXCLUDE"]))
        cmd(f"CccFlipLoose{SUFFIX[m]}", int(flips.loc["EXCLUDE", "INCLUDE"]))
    # Every change of the INCLUDE share is negative, so the maximum is the
    # change closest to zero.
    cmd("CccDeltaSmall", pct(t3.include_delta.max()))
    cmd("CccDeltaLarge", pct(t3.include_delta.min()))

    # The label shares for the other concepts.
    x = orig[(orig.construct == "terrorism") & (orig.arm == "gemma4_31b")]
    cmd("ShareIndTerrGemmaLo", pct(x.INDETERMINATE.min()))
    cmd("ShareIndTerrGemmaHi", pct(x.INDETERMINATE.max()))
    # Within one model, the largest difference between the INCLUDE shares of
    # two definitions, from the rounded shares in tab_shares.tex.
    for c, part in (("assassination", "Assn"), ("terrorism", "Terr"),
                    ("climate_contrarian", "Clim")):
        s = orig[orig.construct == c]
        span = max(float(pct(g.INCLUDE.max())) - float(pct(g.INCLUDE.min()))
                   for _, g in s.groupby("arm"))
        cmd(f"ShareIncSpan{part}", f"{span:.1f}")
    low = orig[orig.construct.isin(["hate_speech", "fake_news"])]
    cmd("ShareIncLowHi", pct(low.INCLUDE.max()))

    # The repeat batches and the re-run on a second platform --------------
    t7 = pd.read_csv(TABLES / "t7_repeat.csv")

    def repeat(df):
        """The number of labels compared and the number that stayed the same."""
        return int(df.n_compared.sum()), int(df.n_identical.sum())

    protest_assn = t7[t7.construct.isin(["protest", "assassination"])]
    n, same = repeat(protest_assn)
    cmd("RepeatFloorSonnet", f"{100 * (n - same) / n:.1f}")
    cmd("RepeatAgrSonnet", f"{100 * same / n:.1f}")
    cmd("RepeatNSonnet", thou(n))
    cmd("RepeatChunkWordSonnet", WORDS[len(protest_assn)])
    # Sonnet's repeat floor, paraphrase floor and definition sensitivity for
    # protest and for assassination.
    for c, repeat_name, name in (("protest", "RepeatFloorProtestSonnet", "Sonnet"),
                                 ("assassination", "RepeatFloorAssnSonnet",
                                  "AssnSonnet")):
        n, same = repeat(protest_assn[protest_assn.construct == c])
        cmd(repeat_name, f"{100 * (n - same) / n:.1f}")
        cmd(f"Floor{name}", pct(row("claude_sonnet5", c).paraphrase_floor))
        cmd(f"Trans{name}", pct(row("claude_sonnet5", c).transmission))
    # Appendix A reports the repeat batches of terrorism, hate speech and
    # climate contrarianism together, and Section 4 reports those of fake news.
    later = t7[~t7.construct.isin(["protest", "assassination", "fake_news"])]
    n, same = repeat(later)
    cmd("RepeatBreadthOkAgr", f"{100 * same / n:.1f}")
    cmd("RepeatBreadthOkN", thou(n))
    cmd("RepeatFakeN", thou(repeat(t7[t7.construct == "fake_news"])[0]))

    t6 = pd.read_csv(TABLES / "t6_crosssite.csv")
    for m in ("gemma4_31b", "qwen36_27b"):
        n, same = repeat(t6[t6.arm == m])
        cmd(f"XSiteAgr{SUFFIX[m]}", f"{100 * same / n:.2f}")

    # Other numbers of Appendix A -----------------------------------------
    # With the original wording of ACLED, Sonnet labelled the texts partly at
    # one reasoning effort and partly at another. Compare the mean
    # disagreement rate of the two pairs with this wording against the rate
    # of the pair of the paraphrases.
    t8 = pd.read_csv(TABLES / "t8_effort.csv")
    cmd("EffortMixedPct", f"{100 * t8[t8.straddles_split].disagreement.mean():.1f}")
    cmd("EffortCleanPct", f"{100 * t8[~t8.straddles_split].disagreement.mean():.1f}")

    # Qwen's first run on the other four concepts, against the main
    # annotation run.
    first_run = sorted(LABELS_QWEN_FIRST_RUN.glob("*.jsonl"))
    n = same = 0
    for f in first_run:
        a, b = verdicts(f), verdicts(LABELS / f.name)
        common = a.keys() & b.keys()
        n += len(common)
        same += sum(a[d] == b[d] for d in common)
    cmd("RerunN", thou(n))
    cmd("RerunDisagreePct", f"{100 * (n - same) / n:.2f}")
    t2_first = pd.read_csv(LABELS_QWEN_FIRST_RUN / "t2_superseded.csv")
    j = t2_first.merge(t2, on=["arm", "construct"], suffixes=("_first", ""))
    j = j[(j.arm == "qwen36_27b") & j.construct.isin(LATER_CONCEPTS)]
    shift = 100 * (j.transmission - j.transmission_first).abs().max()
    # Rounded up, because the paper reports it as an upper bound.
    cmd("RerunMaxShift", f"{math.ceil(shift * 10) / 10:.1f}")
    first_answers = pd.read_csv(
        LABELS_QWEN_FIRST_RUN / "g2_qwen36_27b__BANKED-json_schema.csv")
    cmd("GTwoBreadthQwenFirst", comprehension_score(first_answers, definitions)[1])

    # The missing labels among the analysed texts.
    missing = t1[t1.n_missing > 0]
    if set(missing.arm) - {"qwen36_27b"}:
        stop("Appendix A says that only Qwen has missing labels")
    cmd("MissingN", int(missing.n_missing.sum()))
    cmd("MissingPrompts", len(missing))

    vr =subprocess.run(
        [sys.executable, Path(__file__).with_name("verify_results.py")],
        capture_output=True, text=True)
    if vr.returncode != 0:
        stop("verify_results.py failed")
    cmd("VerifierChecks", vr.stdout.count("PASS  "))

    PAPER.mkdir(parents=True, exist_ok=True)
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n")

    # The tables -----------------------------------------------------------
    groups = []
    for c in ORDER:
        rows = []
        for k, m in enumerate(MODELS):
            r = row(m, c)
            rows.append(
                f"{CONCEPT_NAME[c] if k == 0 else ''} & {MODEL_NAME[m]} & "
                f"{pct(r.transmission)} [{pct(r.transmission_lo)}, "
                f"{pct(r.transmission_hi)}] & "
                f"{pct(r.paraphrase_floor)} [{pct(r.floor_lo)}, "
                f"{pct(r.floor_hi)}] & "
                f"{pct(r.gap_points)} [{pct(r.gap_lo)}, {pct(r.gap_hi)}] & "
                f"{r.ratio:.1f} \\\\")
        groups.append("\n".join(rows))
    (PAPER / "tab_all.tex").write_text(
        "\\setlength{\\tabcolsep}{5pt}%\n"
        "\\begin{tabular}{@{}llrrrr@{}}\n\\toprule\n"
        "Concept & Model & Definition & Paraphrase & & \\\\\n"
        "& & sensitivity & floor & Gap & Ratio \\\\\n"
        "\\midrule\n" + "\n\\addlinespace\n".join(groups) + "\n"
        "\\bottomrule\n\\end{tabular}\n")

    groups = []
    for c in ORDER:
        close, far, w = pairs[c]
        rows = []
        for k, (label, pair) in enumerate((("Closest", close), ("Farthest", far))):
            defs = ", ".join(DEF_NAME[c][d] for d in pair)
            rates = " & ".join(pct(w.loc[pair, m]) for m in MODELS)
            rows.append(f"{CONCEPT_NAME[c] if k == 0 else ''} & {label} & "
                        f"{defs} & {rates} \\\\")
        groups.append("\n".join(rows))
    (PAPER / "tab_pairs.tex").write_text(
        "\\begin{tabular}{@{}ll>{\\raggedright\\arraybackslash}p{5.2cm}rrr@{}}\n"
        "\\toprule\n"
        "Concept & Pair & Definitions & Gemma & Qwen & Sonnet \\\\\n\\midrule\n"
        + "\n\\addlinespace\n".join(groups) + "\n\\bottomrule\n\\end{tabular}\n")

    rows = []
    for c in ORDER:
        s = orig[orig.construct == c]
        cells = []
        for label in ("INCLUDE", "INDETERMINATE"):
            for m in MODELS:
                x = s[s.arm == m][label]
                cells.append(f"{pct(x.min())}--{pct(x.max())}")
        rows.append(f"{CONCEPT_NAME[c]} & " + " & ".join(cells) + " \\\\")
    (PAPER / "tab_shares.tex").write_text(
        "\\setlength{\\tabcolsep}{4pt}%\n"
        "\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n"
        "& \\multicolumn{3}{c}{\\Inc{} share} & "
        "\\multicolumn{3}{c}{\\Indet{} share} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}\n"
        "Concept & Gemma & Qwen & Sonnet & Gemma & Qwen & Sonnet \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    n_macros = sum(ln.startswith("\\newcommand") for ln in lines)
    print(f"[numbers] wrote {n_macros} macros to numbers.tex, and tab_all.tex, "
          f"tab_pairs.tex and tab_shares.tex, to {PAPER}")


if __name__ == "__main__":
    main()
