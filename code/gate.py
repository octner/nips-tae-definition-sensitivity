"""Run the comprehension check with one open-weight model.

In the code, gate and G2 are names of the comprehension check. The script
sends the model the original wording of each definition (impl_0), without a
text to label, and asks questions about what the definition requires. G2_KEY
lists the acceptable answers to each question. These answers are based on the
definition in the prompt, not on the full codebook. Where the definition
permits two answers, G2_KEY accepts both.

The script reads:
- code/annotators.json
- registry/prompt-registry.json and the prompt files in defs/instruments/

It writes these files in out/comprehension_check/:
- g2_<model>.csv, one row for each question and definition
- g2_raw_<model>.jsonl, the raw response for each definition

Run from the repository root, with a vLLM server for the model.
    python3 code/gate.py gemma4_31b
    python3 code/gate.py qwen36_27b
"""

import argparse
import json
import re
import time

import pandas as pd

from _paths import COMPREHENSION, ROOT
import label as L

G2_QUESTIONS = {
    "protest": [
        ("min_participants",
         "What is the minimum number of participants, if any? Answer with the "
         "number, or \"none\" if the definition sets no minimum, or "
         "\"unstated\" if it does not say."),
        ("strikes",
         "Is a labor strike with no accompanying demonstration an event? "
         "Answer \"included\", \"excluded\", or \"unstated\"."),
        ("violence",
         "Is an event in which participants use violence or destroy property "
         "an event under this definition? \"included\", \"excluded\", or "
         "\"unstated\"."),
        ("online",
         "Is an online-only event an event? \"included\", \"excluded\", or "
         "\"unstated\"."),
        ("government_target",
         "Must the action be directed at the government or state? \"yes\", "
         "\"no\", or \"unstated\"."),
        ("party_rally",
         "Is an electoral rally held by political parties in support of their "
         "candidates an event? \"included\", \"excluded\", or \"unstated\"."),
    ],
    "assassination": [
        ("attempts",
         "Does a failed attempt, where the target survives, count as an "
         "event? \"yes\", \"no\", or \"unstated\"."),
        ("state_perpetrator",
         "Does a killing carried out by a state count? \"included\", "
         "\"excluded\", or \"unstated\"."),
        ("prominence",
         "Must the victim be a prominent person or high official? \"yes\", "
         "\"no\", or \"unstated\"."),
        ("custody",
         "Does killing a person held in the perpetrator's custody count? "
         "\"included\", \"excluded\", or \"unstated\"."),
        ("coercion",
         "Is an intention to intimidate or send a message to a wider audience "
         "required? \"yes\", \"no\", or \"unstated\"."),
    ],
    "terrorism": [
        ("state_actor",
         "Can a state be the perpetrator of an event? \"included\", "
         "\"excluded\", or \"unstated\"."),
        ("audience",
         "Is evidence of an intention to coerce, intimidate, or send a "
         "message to an audience beyond the immediate victims required? "
         "\"yes\", \"no\", or \"unstated\"."),
        ("goal",
         "Is a political, economic, religious, or social goal required? "
         "\"yes\", \"no\", or \"unstated\"."),
        ("threat_only",
         "Does a threat of violence, with no physical harm carried out, "
         "count? \"included\", \"excluded\", or \"unstated\"."),
        ("property",
         "Does violence against property alone count? \"included\", "
         "\"excluded\", or \"unstated\"."),
        ("combatants",
         "Do attacks on combatants in the context of legitimate warfare "
         "count? \"included\", \"excluded\", or \"unstated\"."),
    ],
    "hate_speech": [
        ("individual_target",
         "Does an attack on a single named individual, without reference to "
         "a group characteristic, count? \"included\", \"excluded\", or "
         "\"unstated\"."),
        ("protected_list",
         "Is the set of protected characteristics enumerated as a closed, "
         "exhaustive list? \"yes\", \"no\", or \"unstated\"."),
        ("incitement",
         "Must the speech call for violence or discrimination, or can "
         "insulting or demeaning language alone qualify? \"call_required\", "
         "\"insult_suffices\", or \"unstated\"."),
        ("intent",
         "Is the speaker's intent part of the definition? \"yes\", \"no\", "
         "or \"unstated\"."),
        ("reclaimed",
         "Are self-referential or reclaimed uses of slurs addressed as an "
         "exception? \"yes\", \"no\", or \"unstated\"."),
        ("context",
         "Does the verdict depend on context beyond the text itself? "
         "\"yes\", \"no\", or \"unstated\"."),
    ],
    "climate_contrarian": [
        ("warming_denial",
         "Does denying that global warming is happening count? \"included\", "
         "\"excluded\", or \"unstated\"."),
        ("attribution",
         "Does accepting that warming is happening while denying human "
         "causation count? \"included\", \"excluded\", or \"unstated\"."),
        ("impacts",
         "Does downplaying or denying negative impacts count? \"included\", "
         "\"excluded\", or \"unstated\"."),
        ("policy",
         "Do claims that climate policies or solutions are harmful, "
         "unnecessary, or ineffective count? \"included\", \"excluded\", or "
         "\"unstated\"."),
        ("science_attack",
         "Do attacks on climate science or scientists as unreliable count? "
         "\"included\", \"excluded\", or \"unstated\"."),
    ],
    "fake_news": [
        ("falsity",
         "Must the content be verifiably false? \"yes\", \"no\", or "
         "\"unstated\"."),
        ("intent",
         "Is an intention to deceive required? \"yes\", \"no\", or "
         "\"unstated\"."),
        ("form",
         "Must the content mimic the form or appearance of news media? "
         "\"yes\", \"no\", or \"unstated\"."),
        ("satire",
         "Is satire or parody excluded? \"excluded\", \"included\", or "
         "\"unstated\"."),
        ("motive",
         "Is a financial or political motive part of the definition? "
         "\"yes\", \"no\", or \"unstated\"."),
    ],
}

G2_KEY = {
    ("protest", "acled"): {
        "min_participants": {"3"}, "strikes": {"excluded"},
        "violence": {"excluded"}, "online": {"excluded"},
        "government_target": {"no", "unstated"}, "party_rally": {"unstated"}},
    ("protest", "ccc_2021"): {
        "min_participants": {"none", "unstated"}, "strikes": {"included"},
        "violence": {"included"}, "online": {"included"},
        "government_target": {"no", "unstated"},
        "party_rally": {"included", "unstated"}},
    ("protest", "ccc_2024"): {
        "min_participants": {"none", "unstated"}, "strikes": {"excluded"},
        "violence": {"included"}, "online": {"included"},
        "government_target": {"no", "unstated"},
        "party_rally": {"included", "unstated"}},
    ("protest", "mm"): {
        "min_participants": {"50"}, "strikes": {"excluded"},
        "violence": {"unstated"}, "online": {"unstated"},
        "government_target": {"yes"}, "party_rally": {"excluded"}},
    ("protest", "cnts_agd"): {
        "min_participants": {"100"}, "strikes": {"excluded"},
        "violence": {"excluded"}, "online": {"excluded", "unstated"},
        "government_target": {"yes"},
        # G2_KEY accepts "excluded" because of the anti-government purpose
        # that the definition states, and "unstated" because the definition
        # does not name rallies.
        "party_rally": {"excluded", "unstated"}},
    # In gtd and gtd_success, the preamble names "fear, coercion, or
    # intimidation", so G2_KEY accepts "yes" for coercion. But coercion is
    # only one of three criteria, and an event needs at least two, so G2_KEY
    # also accepts "no".
    ("assassination", "gtd"): {
        "attempts": {"yes"}, "state_perpetrator": {"excluded"},
        "prominence": {"yes"}, "custody": {"included", "unstated"},
        "coercion": {"yes", "no"}},
    ("assassination", "gtd_success"): {
        "attempts": {"no"}, "state_perpetrator": {"excluded"},
        "prominence": {"yes"}, "custody": {"included", "unstated"},
        "coercion": {"yes", "no"}},
    ("assassination", "cnts"): {
        "attempts": {"yes"}, "state_perpetrator": {"included", "unstated"},
        "prominence": {"yes"}, "custody": {"included", "unstated"},
        "coercion": {"no", "unstated"}},
    ("assassination", "targeted_killing"): {
        "attempts": {"yes", "unstated"}, "state_perpetrator": {"included"},
        "prominence": {"no"}, "custody": {"excluded"},
        "coercion": {"no", "unstated"}},
    # In gtd_crit, the questions audience, goal and combatants are about the
    # three criteria. An event needs at least two of these criteria, so G2_KEY
    # accepts two answers for each question.
    ("terrorism", "gtd_crit"): {
        "state_actor": {"excluded"}, "audience": {"yes", "no"},
        "goal": {"yes", "no"}, "threat_only": {"included"},
        "property": {"included"}, "combatants": {"excluded", "included"}},
    # Point 2 of the definition names illegal state repression as a context,
    # so a state can be the perpetrator. The definition says "direct violent
    # action" and does not name threats. The victims are "mainly" or "usually"
    # civilians, so the definition does not fully exclude combatants.
    ("terrorism", "schmid"): {
        "state_actor": {"included"}, "audience": {"yes"},
        "goal": {"yes"}, "threat_only": {"excluded", "unstated"},
        "property": {"unstated"}, "combatants": {"excluded", "unstated"}},
    ("terrorism", "enders_sandler"): {
        "state_actor": {"excluded"}, "audience": {"yes"},
        "goal": {"yes"}, "threat_only": {"included"},
        "property": {"unstated"}, "combatants": {"excluded"}},
    # The definition states that it does not mention the combatant and
    # wider-audience distinctions.
    ("terrorism", "weinberg"): {
        "state_actor": {"unstated"}, "audience": {"no", "unstated"},
        "goal": {"yes"}, "threat_only": {"included"},
        "property": {"unstated"}, "combatants": {"included", "unstated"}},
    # The clause about a "qualitatively different manner" means that the label
    # of a text depends on context. The source paper addresses reclaimed use,
    # but the prompt does not, so G2_KEY also accepts "unstated" for reclaimed.
    ("hate_speech", "davidson"): {
        "individual_target": {"excluded", "unstated"},
        "protected_list": {"no", "unstated"},
        "incitement": {"insult_suffices"}, "intent": {"yes"},
        "reclaimed": {"yes", "unstated"}, "context": {"yes", "unstated"}},
    ("hate_speech", "waseem_hovy"): {
        "individual_target": {"excluded", "unstated"},
        "protected_list": {"no", "unstated"},
        "incitement": {"insult_suffices"}, "intent": {"no", "unstated"},
        "reclaimed": {"no", "unstated"}, "context": {"yes", "unstated"}},
    # The definition lists "attributes such as race, ...", and "such as" marks
    # an open list.
    ("hate_speech", "founta"): {
        "individual_target": {"included"}, "protected_list": {"no"},
        "incitement": {"insult_suffices"}, "intent": {"yes"},
        "reclaimed": {"no", "unstated"}, "context": {"no", "unstated"}},
    ("hate_speech", "degibert"): {
        "individual_target": {"excluded"},
        "protected_list": {"no", "unstated"},
        "incitement": {"insult_suffices", "unstated"}, "intent": {"yes"},
        "reclaimed": {"no", "unstated"}, "context": {"no", "unstated"}},
    ("climate_contrarian", "cards"): {
        "warming_denial": {"included"}, "attribution": {"included"},
        "impacts": {"included"}, "policy": {"included"},
        "science_attack": {"included"}},
    # The rubric is about the cause of warming. A denial of the trend can count
    # as a rejection of human-caused warming, or the definition does not name
    # this denial.
    ("climate_contrarian", "cook"): {
        "warming_denial": {"included", "unstated"},
        "attribution": {"included"}, "impacts": {"excluded", "unstated"},
        "policy": {"excluded", "unstated"},
        "science_attack": {"excluded", "unstated"}},
    ("climate_contrarian", "capstick"): {
        "warming_denial": {"included"},
        "attribution": {"included", "unstated"},
        "impacts": {"included", "unstated"}, "policy": {"included"},
        "science_attack": {"included", "unstated"}},
    ("climate_contrarian", "rahmstorf"): {
        "warming_denial": {"included"}, "attribution": {"included"},
        "impacts": {"included"}, "policy": {"excluded", "unstated"},
        "science_attack": {"excluded", "unstated"}},
    # "Fabricated" means that the content is false. The intention to deceive is
    # a criterion of disinformation, which only overlaps with fake news, so
    # intent is not required.
    ("fake_news", "lazer"): {
        "falsity": {"yes"}, "intent": {"no", "unstated"},
        "form": {"yes"}, "satire": {"unstated"},
        "motive": {"no", "unstated"}},
    # The definition excludes satire that readers are unlikely to misunderstand
    # (item 4), but another sentence includes satire that readers could
    # misunderstand.
    ("fake_news", "allcott"): {
        "falsity": {"yes"}, "intent": {"yes"}, "form": {"yes"},
        "satire": {"excluded", "included"}, "motive": {"no", "unstated"}},
    # The definition says "(typically) false or misleading", so falsity is not
    # strictly required.
    ("fake_news", "gelfert"): {
        "falsity": {"no"}, "intent": {"yes"}, "form": {"yes"},
        "satire": {"unstated"}, "motive": {"no", "unstated"}},
    ("fake_news", "zhou"): {
        "falsity": {"yes"}, "intent": {"no"}, "form": {"no"},
        "satire": {"unstated"}, "motive": {"no", "unstated"}},
}

G2_PROMPT = """Below is a definition of an event category, used by a data \
project to decide which events it records.

<DEFINITION>
{definition}
</DEFINITION>

Answer the following questions about THIS DEFINITION AS WRITTEN. Do not use \
any other definition of this concept you may know; if the text above does not \
state the answer, say "unstated".

{questions}

Respond with ONLY a JSON object mapping each question id to your answer:
{{{example}}}"""

# Hate speech, climate contrarianism and fake news are not event categories.
# So G2_PROMPT_X asks about a category of items in place of an event category.
G2_PROMPT_X = """Below is a definition of a category, used by a data \
project to decide which items it records.

<DEFINITION>
{definition}
</DEFINITION>

Answer the following questions about THIS DEFINITION AS WRITTEN. Do not use \
any other definition of this concept you may know; if the text above does not \
state the answer, say "unstated".

{questions}

Respond with ONLY a JSON object mapping each question id to your answer:
{{{example}}}"""


def _g2_prompt_for(construct):
    return G2_PROMPT_X if construct in L.TEXT_CONSTRUCTS else G2_PROMPT


def g2(model):
    """Run the comprehension check for one model and write its answers."""
    rt = L.runtime(model)
    rows, raw_log = [], []
    for e in L.read_registry():
        if e["impl_id"] != "impl_0":
            continue
        qs = G2_QUESTIONS[e["construct"]]
        key = G2_KEY[(e["construct"], e["def_id"])]
        definition = (ROOT / e["file"]).read_text()
        qtext = "\n".join(f"- {qid}: {q}" for qid, q in qs)
        example = ('"reasoning": "<think briefly here>", '
                   + ", ".join(f'"{qid}": "..."' for qid, _ in qs))
        prompt = _g2_prompt_for(e["construct"]).format(
            definition=definition, questions=qtext, example=example)
        fmt = {"type": "json_object"}
        kwargs = dict(model=rt["model"],
                      messages=[{"role": "user", "content": prompt}],
                      response_format=fmt, **rt["params"])
        # The answer starts with a reasoning field, so it needs more tokens
        # than a label.
        kwargs["max_tokens"] = 4096
        if rt["extra_body"]:
            kwargs["extra_body"] = rt["extra_body"]
        ans, err, raw = {}, "", ""
        for attempt in range(3):
            try:
                resp = rt["client"].chat.completions.create(**kwargs)
                raw = resp.choices[0].message.content
                m = re.search(r"\{.*\}", raw or "", re.S)
                if m is None:
                    raise ValueError("the response has no JSON object")
                ans = json.loads(m.group())
                break
            except Exception as ex:
                err = f"{type(ex).__name__}: {ex}"[:120]
                time.sleep(2 ** attempt)
        raw_log.append({"construct": e["construct"], "def_id": e["def_id"],
                        "impl_id": e["impl_id"], "fmt": fmt["type"],
                        "err": err, "raw": raw})
        for qid, _ in qs:
            got = str(ans.get(qid, "")).strip().lower().rstrip(".")
            got = re.sub(r"^\"|\"$", "", got)
            rows.append({"model_id": rt["model"], "construct": e["construct"],
                         "def_id": e["def_id"], "impl_id": e["impl_id"],
                         "question": qid, "answer": got or f"API_FAIL {err}",
                         "expected": "|".join(sorted(key[qid])),
                         "ok": got in key[qid]})
    df = pd.DataFrame(rows)
    COMPREHENSION.mkdir(parents=True, exist_ok=True)
    out = COMPREHENSION / f"g2_{model}.csv"
    df.to_csv(out, index=False)
    with (COMPREHENSION / f"g2_raw_{model}.jsonl").open("w") as fh:
        for r in raw_log:
            fh.write(json.dumps(r) + "\n")
    print(f"[g2] {model}: {int(df.ok.sum())} of {len(df)} answers accepted. "
          f"Wrote {out}.")
    for r in df[~df.ok].itertuples():
        print(f"  {r.construct}/{r.def_id} {r.question}: got {r.answer!r}, "
              f"accepted {r.expected}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Run the comprehension check for one open-weight model.")
    ap.add_argument("model", help="A vLLM model in annotators.json, for "
                                  f"example {L.VLLM_MODELS[0]}.")
    g2(ap.parse_args().model)
