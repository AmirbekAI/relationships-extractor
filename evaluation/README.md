# Evaluation

This is how we measure whether the extraction pipeline and the entity
resolver are actually doing what they're supposed to. It's deliberately
small — the point is the methodology, not the gold-set size.

## Run it

```bash
.venv/bin/python -m evaluation
# or one suite at a time
.venv/bin/python -m evaluation --only extractor
.venv/bin/python -m evaluation --only resolver
```

`OPENAI_API_KEY` is read from `.env`. The resolver eval uses an ephemeral
SQLite database (cleaned up on exit) so your real Postgres is never
touched.

## What gets measured

### 1. Extraction quality — `eval_extractor.py`

For each hand-labelled article in [`gold/articles.json`](gold/articles.json):

  * Run the real `LLMExtractor` on the fixture body (no crawl, no HTTP).
  * Score predicted people against gold (set-based P/R/F1 on names
    normalised by the same `normalize()` the resolver uses, so the eval
    can't drift from the resolver).
  * Score predicted edges against gold under **two** matching rules:
    - **strict** — predicted `relation_type` must equal one of the gold's
      `type_keywords` exactly.
    - **fuzzy** — any gold keyword must appear as a substring of the
      predicted `relation_type` (case-insensitive).

We report both. Strict is the pessimistic floor (it punishes the
open-vocabulary nature of the LLM); fuzzy is the synonym-tolerant
ceiling. The truth lives between them, and the gap tells you how much
"the model gets the relationship right but uses a different word."

### 2. Entity-resolver quality — `eval_resolver.py`

For each pair in [`gold/alias_pairs.json`](gold/alias_pairs.json) (one
`surface` form, the `expected` canonical name, and the
`expected_stage`):

  * Seed a fresh DB with the canonical people + initial aliases.
  * Call `resolve_person(surface, repo, extractor)`.
  * Sniff the resolver's own debug logs to detect which pipeline stage
    fired (`alias` / `levenshtein` / `llm` / `none`).
  * Report **name accuracy** (did we get the right canonical?) and
    **stage accuracy** (did the right resolver tier fire?).

Stage accuracy matters because two failure modes look the same at the
name layer but mean very different things: the LLM stage masking a bug
in Levenshtein is invisible to a pure-accuracy metric.

## Why these choices

* **Fixtures, not live URLs.** Article bodies are frozen in
  `gold/articles.json`. The LLM is still real and nondeterministic
  (that's what we're measuring), but the *input* is identical every
  run, so a drop in F1 reflects a model or prompt regression — not a
  TechCrunch layout change.
* **Open-vocabulary relations.** The extractor prompt lets the LLM
  pick the verb phrase (`criticizes`, `attacks`, `slammed`). Locking
  it to a closed enum would simplify scoring but throw away
  expressiveness. The fuzzy-keyword match gives us a fair score
  without that trade-off.
* **`type_keywords`, plural.** Each gold edge lists the synonyms we
  consider correct (e.g. `["criticize", "criticise"]`). Editing this
  list is how you tune the eval's tolerance — no code change needed.
* **Resolver eval seeded with the canonical names only.** No exotic
  aliases pre-loaded, so the resolver actually has to do work for the
  Levenshtein / LLM cases.

## How to extend

* **More articles.** Append objects to `gold/articles.json` with the
  same shape. Pick articles that probe specific behaviours (sentiment
  in journalist edges, multi-hop relations, ambiguous pronouns, etc.).
* **More relation synonyms.** Just edit `type_keywords` on the
  relevant gold edge — no code change.
* **More resolver cases.** Append to `gold/alias_pairs.json`. The
  `expected_stage` field is what makes this a *resolver* eval and not
  just a name-matching test.

## Known gaps

* **No LLM-as-judge fallback.** When the LLM uses a verb that's
  semantically right but lexically novel (`disparages` where gold has
  `["criticize"]`), fuzzy match misses it. A judge-LLM pass would
  catch that — left out because it doubles the cost per run and
  changes "deterministic eval" to "stochastic eval." Worth adding
  when the gold set grows past ~20 articles.
* **No caching.** Re-running the eval re-hits the LLM. At 3 articles
  this is fine; at 30+ we'd want to cache by `hash(body + prompt +
  model)`.
* **Resolver eval doesn't measure latency.** Useful when the alias
  table grows large — Levenshtein is O(N·m) per call.
