# RelationshipFinder

Crawl news articles, extract the people mentioned and the relationships
between them with an LLM, store the result as a provenance-tracked graph,
and expose it over a small read API.

```
TechCrunch  ──►  Crawler  ──►  LLM extractor  ──►  Resolver  ──►  Graph DB  ──►  REST
```

---

## Table of contents

1. [What it does](#what-it-does)
2. [Quick start](#quick-start)
3. [Architecture](#architecture)
4. [Data model](#data-model)
5. [HTTP API](#http-api)
6. [Single-article pipeline](#single-article-pipeline)
7. [Parallel rescan](#parallel-rescan)
8. [Article lifecycle & crash recovery](#article-lifecycle--crash-recovery)
9. [Entity resolution](#entity-resolution)
10. [Mid-article ambiguity handling](#mid-article-ambiguity-handling)
11. [Extractor abstraction](#extractor-abstraction)
12. [Configuration](#configuration)
13. [Testing](#testing)
14. [Evaluation](#evaluation)

For design rationale and tradeoffs see [`DESIGN.md`](./DESIGN.md).

---

## What it does

Four HTTP endpoints, one job each:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/articles` | Crawl, extract, persist one article URL |
| `POST` | `/rescan` | Walk listing pages across all crawlers and process every URL found |
| `GET` | `/people` | Paginated list of all people in the graph |
| `GET` | `/people/{id}` | One person + every relationship they touch, with article provenance |
| `GET` | `/health` | Liveness probe |

Behind those endpoints, every relationship written to the graph is anchored
to the article and the verbatim sentence that justified it.

---

## Quick start

**Requirements**: Python 3.12+, PostgreSQL 14+ (or SQLite for evaluation),
an OpenAI API key.

```bash
# 1. clone and set up the venv
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. configure
cp .env.example .env
# edit .env — set OPENAI_API_KEY and (optionally) DATABASE_URL

# 3. start Postgres locally, e.g.
docker run -d -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=relationship_finder \
  postgres:16

# 4. run the app
.venv/bin/uvicorn main:app --reload

# 5. submit an article
curl -X POST http://localhost:8000/articles \
  -H 'Content-Type: application/json' \
  -d '{"url": "https://techcrunch.com/<some-article>/"}'

# 6. read the graph
curl http://localhost:8000/people
curl http://localhost:8000/people/<id>
```

Tables are created automatically on first boot, and an idempotent migration
hook adds the chunk-checkpoint columns to pre-existing DBs.

---

## Architecture

![Top-level architecture](./diagrams/top-level-arch.png)

The layers, top to bottom:

- **FastAPI routes** translate HTTP into Pydantic DTOs and back. They hold no
  business logic; they only call the service and pack the response.
- **Dependency injection** (`Depends(get_graph_service)`) hands routes a
  single `GraphService` instance constructed once in the app's lifespan
  handler in [`main.py`](./main.py).
- **`GraphService`** is the only place that opens DB sessions, calls the
  crawler, drives the extractor, and runs the resolver. Every public method
  starts and ends its own transaction.
- **`entity_resolver`** is a pure function module — it maps raw extracted
  names to existing `Person` ids (or signals "create a new one"). It reads
  through the repository; it never opens its own session.
- **`LLMExtractor`** chunks article bodies, calls the configured LLM client
  per chunk, deduplicates results, and runs them through the org filter
  before returning.
- **`BaseLLMClient`** has two concrete implementations: `OpenAIClient`
  (Responses API + Pydantic `text_format`) and `LocalModelClient` (any
  OpenAI-compatible HTTP endpoint, schema-in-prompt strategy).
- **`CrawlerRegistry`** maps URL hostnames to the right `BaseCrawler`
  subclass. `TechCrunchCrawler` is the only one shipped.
- **`GraphRepository`** is the only place that issues SQL. It works against
  whatever async SQLAlchemy session is passed in.

---

## Data model

![Data model](./diagrams/data-model.png)

Five tables, all defined in [`app/db/models.py`](./app/db/models.py):

| Table | What it stores |
|---|---|
| `people` | Canonical persons — one row per real-world individual. |
| `aliases` | Every normalised surface form seen in articles, FK to `people`. `surface_form` is `UNIQUE` across the table. |
| `articles` | One row per processed URL, plus the checkpoint fields (`body_hash`, `sentences_per_chunk`, `total_chunks`, `chunks_processed`). |
| `relationships` | Directed typed edges between two `people`. `(source, target, relation_type)` is `UNIQUE`. |
| `provenance` | Joins a `relationship` to the `article` + quote that justifies it. `(relationship_id, article_id)` is `UNIQUE`. |

All UUID primary keys, all `DateTime` columns are timezone-aware.

---

## HTTP API

### `POST /articles`

Submit a single URL.

```jsonc
// request
{
  "url": "https://techcrunch.com/2024/01/15/example/",
  "sentences_per_chunk": 5    // optional override
}

// response
{
  "article_id": "…",
  "url": "https://techcrunch.com/2024/01/15/example/",
  "title": "Sam Altman Returns to OpenAI as CEO",
  "people_found": 5,
  "relationships_found": 4,
  "status": "processed"        // or "already_exists"
}
```

`400` if no crawler recognises the URL's hostname.
`502` if the crawler reaches the page but can't parse it.

### `POST /rescan`

Walk listing pages across every registered crawler.

```jsonc
// request
{
  "pages": 3,                  // 1..20, default 2
  "sentences_per_chunk": 5,    // optional
  "max_parallel": 4            // optional override of the Settings default
}

// response
{
  "pages_crawled": 3,
  "articles_processed": 22,
  "articles_skipped": 5,       // already_exists hits
  "new_people": 14,
  "new_relationships": 31,
  "status": "complete"         // or "partial" if any URL errored
}
```

### `GET /people?page=&page_size=`

Paginated.

```jsonc
{
  "items": [
    { "id": "…", "canonical_name": "Sam Altman", "aliases": ["sam altman", "altman"] }
  ],
  "total": 142,
  "page": 1,
  "page_size": 20,
  "total_pages": 8
}
```

### `GET /people/{id}`

```jsonc
{
  "id": "…",
  "canonical_name": "Sam Altman",
  "aliases": ["sam altman", "altman", "sam altmann"],
  "bio": null,
  "relationships": [
    {
      "id": "…",
      "source_person_id": "…",  "source_person_name": "Elon Musk",
      "target_person_id": "…",  "target_person_name": "Sam Altman",
      "relation_type": "criticizes",
      "explanation": "Musk publicly criticized the board's decision to remove Altman.",
      "provenance": [
        {
          "article_id": "…",
          "article_url": "https://techcrunch.com/2024/01/15/example/",
          "article_title": "Sam Altman Returns to OpenAI as CEO",
          "quote": "Elon Musk, an early investor and co-founder of OpenAI, publicly criticized the board's decision to remove Altman."
        }
      ]
    }
  ]
}
```

---

## Single-article pipeline

![Single-article pipeline](./diagrams/single-article-pipeline.png)

`GraphService.process_article(url)`:

1. **Pick a crawler** for the URL's hostname; fetch the article (HTML →
   `ArticleContent`).
2. **Hash the body** (SHA-256) for later change detection, and split it into
   sentence chunks of `sentences_per_chunk`.
3. **Open a setup transaction**:
   - First-time article → insert the row with `body_hash`, `total_chunks`,
     `chunks_processed = 0`.
   - Existing row with matching `body_hash` and `chunks_processed > 0` →
     resume at that chunk index.
   - Existing row with different `body_hash` → wipe this article's
     provenance, delete any relationships that lose all provenance, and
     reset the checkpoint.
   - Existing row with `chunks_processed == total_chunks` → return
     `{"status": "already_exists"}` without calling the extractor.
4. **Seed the per-article recency state** from the current alias snapshot
   (used by the resolver to disambiguate first-name-only mentions).
5. **Per-chunk loop**, each chunk in its **own** transaction:
   - Call `extract_one_chunk` — one LLM round-trip.
   - For each extracted person, resolve to a `person_id` (creating a new
     `Person` row if needed) and update the in-memory recency map.
   - For each extracted relationship, upsert the `(src, tgt, type)` edge and
     add a `Provenance` row pointing at this article + the supporting quote.
   - Bump `chunks_processed` in the same transaction.
   - If extraction fails on this chunk: stop, leave `chunks_processed`
     pointing here, the next call resumes from this exact chunk.

The pipeline is structured so that **commits are aligned with progress** —
crash anywhere, the next run picks up where it left off.

---

## Parallel rescan

![Parallel rescan](./diagrams/rescan.png)

`GraphService.rescan(pages, max_parallel)`:

1. **Listing pass** (sequential per crawler): for each registered crawler,
   for each page 1..N, call `get_article_urls(page)` and accumulate into a
   flat list.
2. **Article pass** (concurrent): `asyncio.gather` of one coroutine per URL,
   each wrapped in `async with asyncio.Semaphore(max_parallel)`.
   - Inside each coroutine, `process_article` runs the full pipeline above.
   - Per-article errors are caught and recorded in `summary["errors"]`;
     they do not sink the batch.
3. **Per-host politeness** is enforced inside the crawler itself. Each
   `BaseCrawler` instance owns an `asyncio.Lock`. `TechCrunchCrawler` holds
   it across `await asyncio.sleep(request_delay)` and the HTTP `GET`, so
   no matter how many articles are in flight upstream, fetches to one host
   are serialised with the polite delay between them.

`max_parallel` defaults to `Settings.max_parallel_articles` (default `1` —
old behaviour). Set it higher to overlap LLM extraction + DB writes while
the host lock still rate-limits actual scraping.

---

## Article lifecycle & crash recovery

![Article lifecycle](./diagrams/article-life-cycle.png)

Each `Article` row carries four checkpoint fields:

| Column | Meaning |
|---|---|
| `body_hash` | SHA-256 of `body_text` at first chunking. |
| `sentences_per_chunk` | Chunk size frozen at first chunking. |
| `total_chunks` | Number of chunks the body splits into. |
| `chunks_processed` | The next chunk index to extract on resume. |

The state transitions:

- **Fresh** → on first call, row is inserted, `chunks_processed = 0`.
- **Processing** → per-chunk loop runs, each successful chunk bumps the
  counter atomically with its own writes.
- **Crashed** → loop exited mid-article; row still has
  `chunks_processed < total_chunks`. Next call resumes at exactly that
  chunk; no chunk is ever extracted twice and no completed chunk is lost.
- **Complete** → `chunks_processed == total_chunks`. Subsequent calls with
  the same body return `already_exists` without any LLM call.
- **Reset** → if a later call detects `body_hash` doesn't match the stored
  one, the article's provenance + orphan relationships are wiped and
  `chunks_processed` resets to 0. Then processing starts fresh.

A pre-existing row from before the checkpoint columns were added (i.e.
`total_chunks IS NULL`) is treated as complete and skipped — delete it
manually to force a re-run.

---

## Entity resolution

![Entity resolver](./diagrams/entity-resolver.png)

`resolve_person(raw_name, repo, extractor, *, recency, use_llm_fallback)`
walks five stages and stops at the first that decides:

1. **Normalise**. Strip honorifics (`Mr`, `Dr`, `Jr`, …), ASCII-fold accents,
   lowercase, collapse whitespace.
2. **Stage 1 — exact alias hit**. Look the normalised form up in the
   `aliases` table. O(1) on the index.
3. **Stage 2 — Levenshtein distance**. Scan every alias and compute the
   similarity `1 - dist / max(len_a, len_b)`. Accept if `≥ 0.80`.
4. **Stage 2.5 — unique sub-name**. Collect aliases whose long-token set
   (tokens of ≥ 4 chars) is a strict superset of the surface's long-token
   set — these are the "Altman" → "Sam Altman" / "Satya" → "Satya Nadella"
   abbreviations. If exactly one person matches, accept.
5. **Stage 2.6 — ambiguous sub-name**. If 2+ people match (e.g. "Anthony"
   when both *Anthony Ha* and *Anthony Garcia* are in the DB), check the
   per-article recency map; if exactly one of the candidates matches a
   recency entry, accept. Otherwise return `None`.
6. **Stage 3 — LLM fallback**. Collect candidates with similarity `≥ 0.50`
   *or* a shared long token, ask the LLM to pick one or return null. If
   `use_llm_fallback` is `False`, skip this stage and return `None`.

Every successful match (Levenshtein, sub-name, or LLM) writes the
normalised surface form back to `aliases` so the next occurrence is an
O(1) Stage 1 hit.

The return value is a typed `ResolveResult { person_id, canonical_name,
stage }` — the explicit `stage` field is what the evaluation harness uses
to score per-tier accuracy.

---

## Mid-article ambiguity handling

![Mid-article collision](./diagrams/mid-article-collision.png)

When the resolver can't decide between two same-token people, it consults
a **per-article recency map** — `token → most-recently-resolved person_id`.
Two structures, both per-article, both initialised in
`GraphService.process_article` and threaded through every resolver call:

- **`token_owners`**: `long_token → set[person_id]`. Seeded from the alias
  snapshot at article start; appended to whenever a person is resolved or
  freshly created.
- **`recency`**: `long_token → person_id`, populated **only** for tokens
  whose owner set is ≥ 2 (i.e. currently contested).

The bookkeeping function `update_token_owners_and_recency` runs after every
successful resolve / create. The moment a second owner is added to a token,
that token transitions from uncontested to contested, and `recency` starts
tracking it from that update onward.

Concrete example, with DB starting with only *Anthony Ha*:

1. Article mentions "Anthony Ha" → alias hit, p1. Token `"anthony"` still
   has one owner. `recency` stays empty.
2. Article introduces "Anthony Garcia" (no DB match) → new `Person` p2.
   `token_owners["anthony"] = {p1, p2}` — contested. `recency["anthony"] =
   p2`.
3. Article mentions just "Anthony" → ambiguous sub-name `{p1, p2}`. Look up
   `recency["anthony"]` → p2. Pick *Anthony Garcia*. **Zero LLM calls.**

If recency can't break the tie, the resolver returns `None` and the caller
creates a fresh `Person` row — preferring a missed merge (recoverable by
later dedupe) over a wrong one (sticky bad data).

---

## Extractor abstraction

![Extractor abstraction](./diagrams/extractor-abstraction.png)

`BaseLLMClient` defines one async method:

```python
async def structured_complete(
    self, system_prompt: str, user_message: str, response_schema: Type[T],
) -> T: ...
```

Two implementations ship:

- **`OpenAIClient`** ([`app/extractors/openai_client.py`](./app/extractors/openai_client.py))
  uses the OpenAI Responses API with `text_format=schema`, letting the
  server enforce the JSON shape and returning the parsed Pydantic instance.

- **`LocalModelClient`** ([`app/extractors/local_client.py`](./app/extractors/local_client.py))
  works with any OpenAI-compatible HTTP endpoint (Ollama, LM Studio,
  llama.cpp, vLLM). It appends the JSON schema to the system prompt, then
  extracts the JSON block from the raw response and validates with
  Pydantic.

`LLMExtractor` ([`app/extractors/llm_extractor.py`](./app/extractors/llm_extractor.py))
is the only consumer of `BaseLLMClient`. It owns:

- `split_chunks(article, n)` — the deterministic sentence-then-chunk
  splitter, exposed so callers can drive chunking + extraction one chunk
  at a time.
- `extract_one_chunk(article, text, idx, total)` — used by the
  checkpointed pipeline.
- `extract(article, n)` — convenience wrapper that loops over chunks for
  one-shot extraction.
- `resolve_alias_with_llm(name, candidates)` — Stage 3 of the resolver
  (LLM fallback).

After every extraction, the result passes through `filter_extraction` in
[`app/extractors/filters.py`](./app/extractors/filters.py), which drops
org-shaped people (`OpenAI`, `Microsoft`, `Acme Inc`) and placeholder
markers (`Unknown`, `Anonymous`, `Staff writer`) and any relationship that
touches them.

---

## Configuration

Every tunable lives in `Settings` ([`app/config.py`](./app/config.py)),
loaded from environment variables or `.env`:

| Env var | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Used by `OpenAIClient`. |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model identifier. |
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/relationship_finder` | Async SQLAlchemy URL. |
| `DEFAULT_PAGES` | `2` | Default `pages` value for `/rescan`. |
| `REQUEST_DELAY` | `1.5` | Seconds the crawler sleeps between fetches to one host. |
| `SENTENCES_PER_CHUNK` | `5` | Default chunk size for extraction. |
| `MAX_PARALLEL_ARTICLES` | `1` | Max concurrent articles during a rescan. |
| `RESOLVER_RECENCY_ENABLED` | `true` | Use recency to break ambiguous sub-name matches. |
| `RESOLVER_LLM_FALLBACK_ENABLED` | `true` | Use the LLM as last-resort resolver. |
| `DEFAULT_PAGE_SIZE` | `20` | Default `page_size` for `/people`. |
| `HOST` | `0.0.0.0` | Uvicorn bind host. |
| `PORT` | `8000` | Uvicorn bind port. |
| `LOG_LEVEL` | `info` | Standard-library logging level. |

`Settings` is cached via `lru_cache` — env reads happen on first use, then
the same instance is reused process-wide.

---

## Testing

```bash
# unit + integration suite (no network)
.venv/bin/pytest tests/ \
  --ignore=tests/test_crawler_live.py \
  --ignore=tests/test_extractor.py

# live tests (need OPENAI_API_KEY + internet)
.venv/bin/pytest tests/test_extractor.py -v -s
.venv/bin/pytest tests/test_crawler_live.py -v -s
```

The tests are organised by layer:

| File | Coverage |
|---|---|
| [`tests/test_api.py`](./tests/test_api.py) | All four routes through FastAPI's `TestClient`, status-code mapping, validation, pagination. |
| [`tests/test_chunk_checkpoint.py`](./tests/test_chunk_checkpoint.py) | Crash-resume, body-hash mismatch reset, already-complete short-circuit. Uses a stub extractor that can be set to crash on a specific chunk. |
| [`tests/test_parallel_rescan.py`](./tests/test_parallel_rescan.py) | Wall-clock speedup proof, per-host lock serialisation, error isolation per URL, Settings-default flow. |
| [`tests/test_resolver.py`](./tests/test_resolver.py) | All five resolver stages including the four sub-name cases and the end-to-end mid-article collision scenario. |
| [`tests/test_filters.py`](./tests/test_filters.py) | Org / placeholder filter — positives, negatives, edge cleanup. |
| [`tests/test_crawler.py`](./tests/test_crawler.py) | HTTP intercepted via `httpx.MockTransport`; listing dedup, selector fallback, body sanitisation. |
| [`tests/test_extractor.py`](./tests/test_extractor.py) | Live OpenAI calls — finds the expected people / relationships in a fixed article, chunked vs single. |
| [`tests/test_crawler_live.py`](./tests/test_crawler_live.py) | Live TechCrunch fetch — manual smoke test the selectors still match the live markup. |

89 unit/integration tests run in ~3 seconds on SQLite.

---

## Evaluation

A small gold-set harness lives in [`evaluation/`](./evaluation/).
Run it with:

```bash
.venv/bin/python -m evaluation                      # both suites
.venv/bin/python -m evaluation --only extractor     # extraction only
.venv/bin/python -m evaluation --only resolver      # resolver only
.venv/bin/python -m evaluation --show-diff          # per-article matched/missed/extra edges
```

It evaluates two things:

- **Extraction quality** — for each hand-labelled article in
  [`evaluation/gold/articles.json`](./evaluation/gold/articles.json), run
  the real `LLMExtractor` on the fixture body and score predicted people +
  edges against the gold under both **strict** (`relation_type` == one of
  the gold's `type_keywords` exactly) and **fuzzy** (any keyword is a
  substring of `relation_type`) matching rules. Reports per-article and
  aggregate P / R / F1.

- **Resolver quality** — for each `(surface, expected_canonical,
  expected_stage)` triple in
  [`evaluation/gold/alias_pairs.json`](./evaluation/gold/alias_pairs.json),
  seed a fresh ephemeral SQLite DB, run `resolve_person`, and report
  **name accuracy** (did we land on the right canonical?) and **stage
  accuracy** (did the right resolver tier fire?).

The resolver eval uses a throwaway SQLite file — your real Postgres is
never touched. See [`evaluation/README.md`](./evaluation/README.md) for
the extended methodology, the deployment-profile table, and known gaps.
