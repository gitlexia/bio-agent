# AMR Research Agent

An LLM agent that answers **antimicrobial-resistance (AMR)** questions by
calling real biological-data sources, then writing an answer in which **every
factual claim is cited back to its evidence**. It ships with a small
benchmarking harness so answer quality can actually be *measured*, not just
eyeballed — and a unit-test suite that runs offline.

Built as a focused prototype to explore how frontier models can turn messy
scientific questions into traceable, trustworthy recommendations.

---

## Why

Large language models are fluent but confidently wrong, which is disqualifying
in a scientific setting. The interesting engineering problem isn't "can the
model talk about biology" — it's **grounding** (answers tied to real data) and
**evaluation** (knowing whether the answers are any good). This project is a
small, honest attempt at both.

## What it does

Given a question like *"What does the `gyrA` gene do and how do mutations cause
resistance?"*, the agent:

1. Decides which data tools it needs and calls them (gene annotation, literature).
2. Reads the structured results.
3. Writes an answer where each claim is tagged with its source — `(MyGene.info)`
   or `(PMID: …)` — or admits when the evidence is thin.

### Example

```
$ python bio_agent.py "What organism is mecA associated with and what resistance does it confer?"

  -> mygene_query({'symbol': 'mecA'})
  -> pubmed_search({'query': 'mecA methicillin resistance Staphylococcus aureus'})

The mecA gene encodes penicillin-binding protein 2a (PBP2a), found in
Staphylococcus aureus (MyGene.info). PBP2a has a low affinity for beta-lactam
antibiotics, which allows the bacterium to continue building its cell wall even
in their presence — the mechanism behind methicillin-resistant S. aureus (MRSA)
(PMID: …). Carriage of mecA is the defining genetic marker used to identify MRSA
isolates (PMID: …).
```

*(Illustrative run. Replace with your own output; PMID citations reflect whatever
PubMed returns live at query time.)*

## How it works

```
question
   │
   ▼
┌──────────────┐   tool calls    ┌──────────────────────────┐
│  agent loop  │ ──────────────▶ │  tools (plain functions) │
│  (LLM +      │ ◀────────────── │   • mygene_query  → MyGene.info
│   tool use)  │   JSON results  │   • pubmed_search → NCBI PubMed
└──────────────┘                 └──────────────────────────┘
   │
   ▼
cited answer
```

- **Tools are plain, independent functions.** Each hits one data source and
  returns JSON. Adding a tool means writing one function and one schema — nothing
  else in the loop changes. They're pure, so they're trivial to unit-test.
- **The agent loop** hands the model its tools, runs whatever it asks for,
  feeds the results back, and repeats until the model produces a final answer
  (bounded by a step limit so it can't run away).
- **The LLM client is created lazily**, so the module imports — and its logic
  tests — without an API key present.
- **Citations are enforced by the system prompt**, and because the model only
  ever sees tool output, it can't cite a source it didn't actually query.

## Quickstart

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

# ask a question
python bio_agent.py "What class of antibiotic does the vanA gene confer resistance to?"

# run the benchmark
python bio_agent.py --eval
```

Set `MODEL` near the top of `bio_agent.py` to a model your account can access
(see https://docs.claude.com/en/docs/about-claude/models). The data sources
(MyGene.info, NCBI PubMed) are free and need no key.

## Tests

The suite uses only the standard library — no extra install, no network, no API
key:

```bash
python -m unittest discover -s tests -v
```

It covers the data tools (with mocked HTTP, including their empty-result paths),
the keyword grader, and the multi-step agent loop (with a faked LLM client).

## Evaluation

A curated question set (`EVAL_SET`) pairs AMR questions with the facts a correct
answer must contain. `run_eval()` runs the full agent on each and scores it:

```
$ python bio_agent.py --eval
[1] PASS  (2/2 keywords)  What organism is the mecA gene associated with ...
[2] PASS  (2/2 keywords)  What does the gyrA gene do and how do mutations ...
...

Score: 5/5
```

The keyword grader is deliberately crude — it's cheap, deterministic, and good
enough to catch regressions while iterating on prompts and tools. Replacing it
with an **LLM-as-judge** grader is the obvious next upgrade.

## Design decisions

- **Grounding over fluency.** The tools exist to stop the model inventing facts;
  citations make it auditable.
- **A real eval from day one.** Even a rough benchmark turns "seems fine" into a
  number you can improve against.
- **Boring, well-established libraries.** Just `requests` and the `anthropic`
  SDK; tests are stdlib-only.
- **Small, testable units.** Tools, the grader, and the agent loop are separated
  so each can be tested in isolation.

## Limitations

A prototype, not a production system: no caching or retry/backoff on the data
APIs, the keyword grader is approximate, there's no persistence, and the eval
set is small. MyGene.info is also richer for eukaryotic genes than for some
bacterial AMR genes, so literature (PubMed) often carries more of the answer.

## Roadmap

- **More tools:** UniProt (proteins) and a small structured gene → mechanism →
  drug knowledge graph for relational lookups.
- **LLM-as-judge** grading for nuance beyond keyword matching.
- **Caching + backoff** on external calls for reliability and cost.
- **Fuller literature grounding:** fetch and summarise abstract text, not just
  metadata.

## Tech stack

Python · Anthropic API (tool use) · MyGene.info · NCBI E-utilities (PubMed)
