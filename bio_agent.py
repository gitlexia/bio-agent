"""
bio_agent.py
============
A small LLM agent that answers antimicrobial-resistance (AMR) bioinformatics
questions by calling REAL biological-data tools, then writes an answer in which
every factual claim is cited back to the evidence it used.

It also ships a tiny benchmarking harness (`--eval`) that scores the agent
against a curated question set -- the "evaluation" half of LLM work that most
quick demos skip.

Design notes
------------
* Tools are plain, independently-testable functions (easy to unit-test and to
  add to). The agent only ever sees their JSON output.
* The LLM client is created lazily, so the module can be imported -- and its
  pure logic unit-tested -- without an API key present.
* The data sources are free and need no API key:
    - MyGene.info  (gene annotation)   https://mygene.info
    - NCBI PubMed  (literature)        https://www.ncbi.nlm.nih.gov/pmc/tools/developers/
* The only secret is your LLM key.

Run
---
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...

    # ask a single question:
    python bio_agent.py "What does the gyrA gene do and how do mutations cause resistance?"

    # run the benchmark:
    python bio_agent.py --eval

Test
----
    python -m unittest discover -s tests -v    # no extra dependencies needed
"""

from __future__ import annotations

import json
import sys
import time

import requests

# Set this to whatever model your account has access to.
# Current model strings are listed at https://docs.claude.com/en/docs/about-claude/models
MODEL = "claude-sonnet-5-5"


# --------------------------------------------------------------------------- #
#  LLM client (lazy)                                                           #
#  Created on first use so the module imports cleanly without an API key,      #
#  which keeps the pure logic below unit-testable.                             #
# --------------------------------------------------------------------------- #
_client = None


def get_client():
    """Return a cached Anthropic client, creating it on first use."""
    global _client
    if _client is None:
        import anthropic  # imported lazily: not needed to test the tools/grader

        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the env
    return _client


# --------------------------------------------------------------------------- #
#  Tools  -- plain functions, each hitting one real biological data source.    #
#  Keeping them pure and independent makes them trivial to unit-test.          #
# --------------------------------------------------------------------------- #
def mygene_query(symbol: str, species: str = "all") -> dict:
    """Look up a gene by symbol via MyGene.info; return its core annotation."""
    resp = requests.get(
        "https://mygene.info/v3/query",
        params={
            "q": f"symbol:{symbol}",
            "species": species,
            "size": 1,
            "fields": "symbol,name,summary,taxid,type_of_gene",
        },
        timeout=20,
    )
    resp.raise_for_status()
    hits = resp.json().get("hits", [])
    return hits[0] if hits else {"error": f"No gene found for symbol {symbol!r}"}


def pubmed_search(query: str, max_results: int = 3) -> list[dict]:
    """Search PubMed and return a few relevant papers (title, journal, year, PMID)."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    ids = (
        requests.get(
            f"{base}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": max_results,
                "sort": "relevance",
            },
            timeout=20,
        )
        .json()["esearchresult"]["idlist"]
    )
    if not ids:
        return []

    time.sleep(0.4)  # be polite to NCBI's rate limits
    summary = requests.get(
        f"{base}/esummary.fcgi",
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
        timeout=20,
    ).json()["result"]

    papers = []
    for pmid in ids:
        rec = summary.get(pmid, {})
        papers.append(
            {
                "pmid": pmid,
                "title": rec.get("title", ""),
                "journal": rec.get("fulljournalname", ""),
                "year": rec.get("pubdate", "")[:4],
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )
    return papers


# Registry: maps the tool name the model calls -> the function that runs it.
TOOLS_IMPL = {
    "mygene_query": mygene_query,
    "pubmed_search": pubmed_search,
}

# Schemas the model sees. Add a new tool by writing a function above and a
# schema here -- nothing else changes.
TOOL_SCHEMAS = [
    {
        "name": "mygene_query",
        "description": (
            "Look up a gene by its symbol (e.g. 'mecA', 'gyrA', 'vanA') and get "
            "its name, a function summary, and the organism it belongs to."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Gene symbol"},
                "species": {
                    "type": "string",
                    "description": "Species name/taxid, or 'all'",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "pubmed_search",
        "description": (
            "Search the PubMed literature database and return a few relevant "
            "papers with title, journal, year, PMID and URL."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Focused search query"},
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
]

SYSTEM = (
    "You are a bioinformatics research assistant specialising in antimicrobial "
    "resistance. Before answering, call the available tools to gather evidence "
    "from real biological databases. Every factual claim in your final answer "
    "MUST cite the evidence that supports it -- e.g. (MyGene.info) or "
    "(PMID: 12345678). If the evidence is insufficient, say so plainly rather "
    "than guessing."
)


# --------------------------------------------------------------------------- #
#  Agent loop                                                                  #
# --------------------------------------------------------------------------- #
def run_agent(question: str, max_steps: int = 5, verbose: bool = True) -> str:
    """Run the tool-using agent until it produces a final (cited) answer."""
    client = get_client()
    messages: list[dict] = [{"role": "user", "content": question}]

    for _ in range(max_steps):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")

        # Execute every tool the model asked for, hand the results back.
        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            if verbose:
                print(f"  -> {block.name}({block.input})")
            try:
                result = TOOLS_IMPL[block.name](**block.input)
            except Exception as exc:  # surface errors to the model, don't crash
                result = {"error": str(exc)}
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result)[:4000],
                }
            )
        messages.append({"role": "user", "content": tool_results})

    return "Stopped: reached max steps without a final answer."


# --------------------------------------------------------------------------- #
#  Benchmarking harness                                                        #
#  A curated eval set + a simple keyword grader. Crude but honest, and it      #
#  gives you a number to iterate against. (Upgrade idea: LLM-as-judge.)        #
# --------------------------------------------------------------------------- #
EVAL_SET = [
    {
        "q": "What organism is the mecA gene associated with and what resistance does it confer?",
        "must_mention": ["staphylococcus", "methicillin"],
    },
    {
        "q": "What does the gyrA gene do and how do mutations in it cause antibiotic resistance?",
        "must_mention": ["gyrase", "fluoroquinolone"],
    },
    {
        "q": "What class of antibiotic does the vanA gene confer resistance to?",
        "must_mention": ["vancomycin"],
    },
    {
        "q": "The blaKPC gene produces an enzyme that breaks down which class of antibiotics?",
        "must_mention": ["carbapenem"],
    },
    {
        "q": "Mutations in rpoB are a common cause of resistance to which TB drug?",
        "must_mention": ["rifampicin"],
    },
]


def score_answer(answer: str, must_mention: list[str]) -> tuple[list[str], bool]:
    """Grade one answer by keyword coverage.

    Returns the list of matched keywords and whether all of them were found.
    Pulled out of run_eval so the grading rule can be unit-tested on its own.
    """
    text = answer.lower()
    hits = [kw for kw in must_mention if kw.lower() in text]
    return hits, len(hits) == len(must_mention)


def run_eval() -> None:
    passed = 0
    for i, case in enumerate(EVAL_SET, 1):
        answer = run_agent(case["q"], verbose=False)
        hits, ok = score_answer(answer, case["must_mention"])
        passed += ok
        print(
            f"[{i}] {'PASS' if ok else 'FAIL'}  "
            f"({len(hits)}/{len(case['must_mention'])} keywords)  {case['q']}"
        )
    print(f"\nScore: {passed}/{len(EVAL_SET)}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--eval":
        run_eval()
    elif len(sys.argv) > 1:
        print(run_agent(" ".join(sys.argv[1:])))
    else:
        print(__doc__)
