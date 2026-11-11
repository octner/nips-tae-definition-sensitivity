"""Ask the open-weight models, Gemma 4 31B and Qwen 3.6 27B, to label each text
with each prompt of its concept.

A prompt pairs one wording of a definition with a query, and each model answers
it with one label, INCLUDE, EXCLUDE or INDETERMINATE. In the code, an
instrument is a prompt. Claude Sonnet 5 ran through an agentic command-line
harness, not through this script.

The script reads:
- code/annotators.json
- registry/prompt-registry.json and the prompt files in defs/instruments/
- the frozen samples, data/sample.parquet and data/sample_expansion.parquet

It writes:
- out/labels/, one results file and one sidecar file for each model and
  prompt, with the command `run`
- registry/prompt-registry.json, with the command `instruments`

A run labels only the texts that are not yet in the results file. The function
check_guard stops the run if the sidecar records the hashes of another prompt
or sample.

Run from the repository root. The command `run` needs a vLLM server for each
model.
    python3 code/label.py instruments
    python3 code/label.py run
"""

import argparse
import functools
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from _paths import (DATA, DEFS, LABELS, REGISTRY, ROOT, VERDICTS,
                    check_guard, instrument_sha, suite_version_for)

INSTR_DIR = DEFS / "instruments"
SAMPLE = DATA / "sample.parquet"
SAMPLE_EXPANSION = DATA / "sample_expansion.parquet"

CONCURRENCY = 8  # requests that are open at the same time
MAX_RETRIES = 5

# The definition in a prompt file may have at most this number of tokens. The
# estimate is 1.5 tokens for each word.
MAX_INSTR_TOKENS = 1200

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string",
                    "enum": ["INCLUDE", "EXCLUDE", "INDETERMINATE"]},
        "confidence": {"type": "string", "enum": ["HIGH", "LOW"]},
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "confidence", "rationale"],
    "additionalProperties": False,
}

# PROMPT is the query for protest, assassination and terrorism. The query is
# the same for each definition of a concept, so the prompts of a concept
# differ only in the definition.
PROMPT ="""You are applying a published event-catalogue definition to a text, \
for a measurement study of coding rules.

Below is the definition. Apply it exactly as written. Where it conflicts with \
any other definition of this concept you may know, the text below wins.

<DEFINITION>
{definition}
</DEFINITION>

Text to judge:
<TEXT>
{unit}
</TEXT>

Respond with ONLY a JSON object:
{{"verdict": "INCLUDE"|"EXCLUDE"|"INDETERMINATE",
  "confidence": "HIGH"|"LOW",
  "rationale": "<one sentence naming the clause of the definition that decided it>"}}

- INCLUDE: the text describes an event that this definition counts.
- EXCLUDE: the text describes no such event, or an event this definition \
excludes.
- INDETERMINATE: the text does not state what is needed to decide under this \
definition (for example, it names no crowd size and the definition requires \
one). INDETERMINATE is a real answer, not a failure; use it whenever the \
deciding fact is absent from the text.
- Judge ONLY from the text. Do not use outside knowledge of the event, the \
place, or the organisations named."""

# Hate speech, climate contrarianism and fake news are not events. So their
# query, PROMPT_X, states the same task and labels for a text in place of an
# event.
PROMPT_X ="""You are applying a published definition to a text, for a \
measurement study of coding rules.

Below is the definition. Apply it exactly as written. Where it conflicts with \
any other definition of this concept you may know, the text below wins.

<DEFINITION>
{definition}
</DEFINITION>

Text to judge:
<TEXT>
{unit}
</TEXT>

Respond with ONLY a JSON object:
{{"verdict": "INCLUDE"|"EXCLUDE"|"INDETERMINATE",
  "confidence": "HIGH"|"LOW",
  "rationale": "<one sentence naming the clause of the definition that decided it>"}}

- INCLUDE: the text is, or describes, an instance of what this definition \
covers.
- EXCLUDE: the text is no such instance, or is an instance this definition \
excludes.
- INDETERMINATE: the text does not state what is needed to decide under this \
definition (for example, the definition requires an intent that the text \
neither states nor rules out). INDETERMINATE is a real answer, not a failure; \
use it whenever the deciding fact is absent from the text.
- Judge ONLY from the text. Do not use outside knowledge of the events, \
people, places, or organisations named."""

TEXT_CONSTRUCTS = {"hate_speech", "climate_contrarian", "fake_news"}


def _prompt_for(construct):
    return PROMPT_X if construct in TEXT_CONSTRUCTS else PROMPT


# --- models ----------------------------------------------------------------

ANNOTATORS = json.loads(
    (Path(__file__).resolve().parent / "annotators.json").read_text())["annotators"]
VLLM_MODELS = [n for n, c in ANNOTATORS.items() if c["provider"] == "vllm"]


def runtime(name):
    """The client, the model name and the request settings of one open-weight
    model."""
    if name not in ANNOTATORS:
        raise SystemExit(f"[label] {name!r} is not in annotators.json. "
                         f"The models are {sorted(ANNOTATORS)}.")
    cfg = ANNOTATORS[name]
    if cfg["provider"] != "vllm":
        raise SystemExit(f"[label] This script does not run {name}. "
                         f"{cfg.get('note', '')}")
    if cfg["response_format"] != "json_schema":
        raise SystemExit(f"[label] {name}: the only response format of this "
                         f"script is json_schema.")
    from openai import OpenAI
    base = os.environ.get(cfg["base_url_env"],
                          f"http://127.0.0.1:{cfg['port']}/v1")
    client = OpenAI(api_key="EMPTY", base_url=base, timeout=600)
    fmt = {"type": "json_schema",
           "json_schema": {"name": "tae_verdict", "schema": SCHEMA}}
    return dict(name=name, client=client, model=cfg["model"],
                params=cfg["params"], response_format=fmt,
                extra_body=cfg.get("extra_body"))


# --- prompts ---------------------------------------------------------------

def _ntokens(text):
    return int(len(text.split()) * 1.5)


def render_instruments():
    """Write the prompt registry, with one record for each prompt file."""
    entries, too_long = [], []
    for p in sorted(INSTR_DIR.glob("*/*__*.md")):
        construct = p.parent.name
        def_id, impl_id = p.stem.split("__", 1)
        text = p.read_text()
        if _ntokens(text) > MAX_INSTR_TOKENS:
            too_long.append(p.name)
        entries.append({
            "construct": construct, "def_id": def_id, "impl_id": impl_id,
            "instrument_sha": instrument_sha(construct, def_id, impl_id, text),
            "file": str(p.relative_to(ROOT)),
            "n_tokens_est": _ntokens(_prompt_for(construct)) + _ntokens(text),
            "suite_version": suite_version_for(construct),
        })
    if not entries:
        raise SystemExit(f"[label] There are no prompt files in {INSTR_DIR}. "
                         f"The expected name is "
                         f"<construct>/<def_id>__<impl_id>.md.")
    if too_long:
        raise SystemExit("[label] These definitions are longer than "
                         f"{MAX_INSTR_TOKENS} tokens: {', '.join(too_long)}")
    REGISTRY.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"[label] wrote {len(entries)} prompts to {REGISTRY}")


def read_registry():
    return json.loads(REGISTRY.read_text())


# --- texts -----------------------------------------------------------------

@functools.cache
def _read_sample(path):
    """A frozen sample and its hash, read only once."""
    if not path.exists():
        raise SystemExit(f"[label] {path.relative_to(ROOT)} is absent. "
                         f"The texts are not part of this repository.")
    df = pd.read_parquet(path)
    return df, df.frame_sha.iloc[0]


def _units_for(construct):
    """The texts of one concept and the hash of their sample.

    Hate speech, climate contrarianism and fake news use the sample from the
    public datasets. Terrorism uses the assassination texts.
    """
    if construct in TEXT_CONSTRUCTS:
        df, frame_sha = _read_sample(SAMPLE_EXPANSION)
        return df[df.construct == construct], frame_sha
    df, frame_sha = _read_sample(SAMPLE)
    rows = "assassination" if construct == "terrorism" else construct
    return df[df.construct == rows], frame_sha


# --- labels ----------------------------------------------------------------

def call(rt, unit_text, doc_id, definition, e, frame_sha):
    """Ask the model for the label of one text with one prompt."""
    prompt = _prompt_for(e["construct"]).format(definition=definition,
                                               unit=unit_text)
    kwargs = dict(model=rt["model"],
                  messages=[{"role": "user", "content": prompt}],
                  response_format=rt["response_format"], **rt["params"])
    if rt["extra_body"]:
        kwargs["extra_body"] = rt["extra_body"]
    for attempt in range(MAX_RETRIES):
        try:
            resp = rt["client"].chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content
            r = json.loads(re.search(r"\{.*\}", raw, re.S).group())
            if r.get("verdict") not in VERDICTS:
                raise ValueError(f"bad verdict {r.get('verdict')!r}")
            break
        except Exception as ex:
            if attempt == MAX_RETRIES - 1:
                r = {"verdict": None, "confidence": "LOW",
                     "rationale": f"API_FAIL {type(ex).__name__}: {ex}"[:200]}
            else:
                time.sleep(2 ** attempt)
    return {"doc_id": doc_id, "verdict": r.get("verdict"),
            "confidence": r.get("confidence", "LOW"),
            "rationale": r.get("rationale", "")[:300],
            "construct": e["construct"], "def_id": e["def_id"],
            "impl_id": e["impl_id"], "model_id": rt["model"], "rep": 0,
            "instrument_sha": e["instrument_sha"], "frame_sha": frame_sha,
            "suite_version": suite_version_for(e["construct"])}


def _done_ids(path):
    if not path.exists():
        return set()
    return {json.loads(l)["doc_id"] for l in path.open() if l.strip()}


def _label(rt, todo, out, e, frame_sha):
    """Label the texts in `todo` and add one line for each to `out`."""
    definition = (ROOT / e["file"]).read_text()
    t0 = time.time()
    with out.open("a") as f, ThreadPoolExecutor(CONCURRENCY) as ex:
        futs = [ex.submit(call, rt, row.unit_text, row.doc_id, definition,
                          e, frame_sha)
                for row in todo.itertuples()]
        for n, fut in enumerate(as_completed(futs), 1):
            f.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
            if n % 100 == 0 or n == len(futs):
                f.flush()
                print(f"  [{rt['name']}] {n}/{len(todo)} "
                      f"({n / (time.time() - t0):.1f} texts/s)", flush=True)


def run(names=None):
    """Label the texts with every prompt of the registry, for each model."""
    for name in names or VLLM_MODELS:
        rt = runtime(name)
        LABELS.mkdir(parents=True, exist_ok=True)
        for e in read_registry():
            units, frame_sha = _units_for(e["construct"])
            out = LABELS / (f"{name}__{e['construct']}__{e['def_id']}"
                            f"__{e['impl_id']}.jsonl")
            check_guard(out, e["instrument_sha"], frame_sha)
            todo = units[~units.doc_id.isin(_done_ids(out))]
            print(f"[{name}] {e['construct']}/{e['def_id']}/{e['impl_id']} "
                  f"(done: {len(units) - len(todo)})  to do: {len(todo)}",
                  flush=True)
            if len(todo):
                _label(rt, todo, out, e, frame_sha)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Label each text under each prompt with a vLLM model.")
    ap.add_argument("cmd", choices=["instruments", "run"],
                    help="`instruments` writes the prompt registry. "
                         "`run` labels the texts.")
    ap.add_argument("models", nargs="*",
                    help="The models for `run`. The default is "
                         f"{' and '.join(VLLM_MODELS)}.")
    a = ap.parse_args()
    if a.cmd == "instruments":
        render_instruments()
    else:
        run(a.models)
