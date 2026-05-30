# Design rationale

This document is the *why* companion to [`README.md`](./README.md). For
every non-obvious choice in the system, what's documented here is:
the alternatives that were on the table, what trade-off the chosen design
favours, and (where applicable) the math that justifies it.

---

## Table of contents

1. [FastAPI + async stack](#1-fastapi--async-stack)
2. [Per-chunk checkpoints](#2-per-chunk-checkpoints)
3. [Multi-stage entity resolution](#3-multi-stage-entity-resolution)
4. [Per-article recency for ambiguous sub-names](#4-per-article-recency-for-ambiguous-sub-names)
5. [Semaphore + per-host lock concurrency](#5-semaphore--per-host-lock-concurrency)
6. [Provenance as a first-class table](#6-provenance-as-a-first-class-table)
7. [`BaseLLMClient` abstraction with two implementations](#7-basellmclient-abstraction-with-two-implementations)
8. [Belt-and-suspenders org / placeholder filter](#8-belt-and-suspenders-org--placeholder-filter)
9. [Evaluation harness with strict + fuzzy scoring](#9-evaluation-harness-with-strict--fuzzy-scoring)
10. [Idempotent column migrations instead of Alembic](#10-idempotent-column-migrations-instead-of-alembic)
11. [Two configurable resolver knobs](#11-two-configurable-resolver-knobs)

---

## 1. FastAPI + async stack

**Choice.** FastAPI for the HTTP surface, `httpx.AsyncClient` for crawling,
`openai.AsyncOpenAI` for LLM calls, `SQLAlchemy 2.0 async` over `asyncpg`
for persistence, `asyncio` everywhere.

**Why.** Every interesting unit of work in this pipeline is **I/O-bound**:
HTTP fetch (TechCrunch), LLM round-trip (OpenAI), SQL commit (Postgres).
Between those waits, the actual CPU work — chunking, parsing, resolver
arithmetic — is trivial. A synchronous stack would spend almost all of its
time blocked on syscalls; a thread-pool would pay context-switch and GIL
costs for nothing, since the GIL is released exactly when we're waiting
on I/O anyway. Asyncio gives us cheap cooperative concurrency at the only
points it matters and stays out of the way otherwise.

**What we gave up.** True CPU parallelism (we don't need it). The
mental-model overhead of "everything is `async`" (paid once, amortised
across the whole codebase).

---

## 2. Per-chunk checkpoints

**Choice.** Split the article body into N sentence-chunks; commit each
chunk's writes plus an updated `articles.chunks_processed` pointer in one
transaction; resume at that pointer on the next call.

**The naive alternative.** Process the whole article in one transaction
(or one logical retry unit). Any failure — rate limit, network blip,
malformed JSON, container restart — rolls everything back; the next
attempt re-extracts from chunk 1.

### Why it matters

Working the expected cost through a geometric retry model (each chunk fails
independently with probability `p`, a failure without checkpoints forces a
restart from chunk 1) shows that the no-checkpoint cost grows
**super-linearly** in the per-chunk failure rate: at low `p` the two designs
are within a few percent, but the gap widens fast as `p` climbs, and beyond
`p ≈ 30%` the no-checkpoint variant approaches "may never finish in a bounded
retry budget." On a realistic rescan (100 articles, busy-hour failure rates)
that's a meaningful recurring difference in LLM spend, plus the wall-clock
cost of replaying already-extracted chunks on every failure.

### What you give up

Schema complexity: four extra columns on `articles`, an idempotent
migration to add them to pre-existing rows, a slightly more involved
"first call vs resume vs already-complete vs body-changed" decision tree
in `process_article`. We pay that complexity once; the savings recur
forever.

A second small cost: on resume after crash, the per-article recency map
starts empty (it lived only in memory). The resolver loses cross-chunk
recency for the resumed chunks — but anything we already wrote to
`aliases` still gets O(1) hits, so the degradation is bounded.

---

## 3. Multi-stage entity resolution

**Choice.** A five-stage pipeline that walks from cheap-and-precise to
expensive-and-lenient, returning at the first stage that decides:

```
exact alias  →  Levenshtein ≥ 0.80  →  unique sub-name
                                    →  ambiguous sub-name + recency
                                    →  LLM fallback
```

**Alternatives considered.**

- **Just exact alias lookup.** Cannot handle typos, honorifics, or
  first-name-only references. Would create separate `Person` rows for
  "Sam Altman", "Sam Altmann", and "Altman".
- **Just an LLM call.** Spends a token round-trip on every single name
  occurrence in every article. At ~10 person-mentions per article and
  ~22 articles per rescan, that's 220 extra LLM calls per rescan — and
  most of them are "Sam Altman" → "Sam Altman", a question the alias
  table answers in one indexed lookup.
- **Just fuzzy matching (Levenshtein everywhere).** Cannot disambiguate
  same-similarity candidates ("Anthony" matches both *Anthony Ha* and
  *Anthony Garcia* equally) and silently merges them.

**Why the funnel works.** Each stage handles a distinct failure mode of
the previous one:

- Stage 1 handles every name we've already seen — the **hot path**.
- Stage 2 handles **typos and minor spelling variations** that don't
  warrant an LLM call.
- Stage 2.5 handles the **introduce-then-shorten** pattern endemic to
  journalism prose ("Sam Altman, CEO of OpenAI, said … Altman
  continued …") deterministically, without an LLM.
- Stage 2.6 handles the **same shortening when multiple people share
  the token**, using local article context.
- Stage 3 catches the **long-tail synonym / nickname cases** the rules
  miss ("OpenAI CEO" → "Sam Altman").

**Side effect that compounds.** Every successful Levenshtein / sub-name /
LLM hit writes the normalised surface form to `aliases`. So a fuzzy match
costs `O(N_aliases)` *once*; from then on the same surface form is a
Stage 1 hit, cost O(1).

---

## 4. Per-article recency for ambiguous sub-names

**Choice.** During an article, maintain two in-memory structures:

- `token_owners: token → set[person_id]` — every long token seen during
  this article, and which people own it. Seeded from the alias snapshot
  at article start, updated whenever a person is resolved or created.
- `recency: token → person_id` — only populated for tokens whose owner
  set has reached 2+, tracking the most-recently-handled owner.

When the resolver hits an ambiguous sub-name match, it looks the
surface's long tokens up in `recency` and accepts the candidate if
exactly one of them is the recency target.

**Alternative 1: refuse on ambiguity, always.** Predecessor design.
Creates a fresh `Person` row for every ambiguous mention. Result: a
re-read of an article that already has *Anthony Ha* in the DB and
introduces a second *Anthony Garcia* spawns a third "Anthony" person —
and every later mention of "Anthony" makes another. Massive duplicates,
all deferred to a separate dedupe pass.

**Alternative 2: snapshot ambiguity at article start.** Compute the set
of contested tokens once at the top of the article and use that for all
disambiguation. Result: a *fresh* collision introduced mid-article — the
classic "article tells you about Anthony Garcia three sentences in" case
— gets missed, because at snapshot time "anthony" had only one owner.
The fix is dynamic detection: the moment a second owner is added during
the article, the token is contested *from that point on*. This is the
distinction the test `test_dynamic_detection_mid_article_collision_…`
pins.

**What we give up.** Recency is local to the article — on crash-resume,
the maps start empty. This is intentional: per-article recency is
domain-justified ("the same article tends to use one canonical form
consistently"), while cross-article recency isn't — *Anthony Ha* in
yesterday's article tells you nothing about *Anthony* in today's.

We also accept the risk of a **wrong merge** in genuinely ambiguous text.
The `RESOLVER_RECENCY_ENABLED` setting lets cost-conscious deployments
revert to the refuse-on-ambiguity policy.

---

## 5. Semaphore + per-host lock concurrency

**Choice.** `rescan` runs articles concurrently under an
`asyncio.Semaphore(max_parallel)`, but each crawler instance holds its own
`asyncio.Lock` across `await asyncio.sleep(request_delay)` and the HTTP
`GET`. So:

- **Upstream** (Semaphore): bounds how many *articles* are in flight at
  once. Raising it parallelises the work that's actually expensive
  (LLM round-trips) and the work that's actually concurrent-safe
  (independent DB transactions).
- **Downstream** (Lock): bounds how many HTTP fetches go to a single
  host. The polite delay is enforced regardless of upstream concurrency.

**The single-knob alternative.** "Just use the Semaphore at N." If
`max_parallel = 4`, four coroutines burst four simultaneous GETs at
TechCrunch every iteration, blowing past the polite-delay contract and
getting us rate-limited or banned. The semaphore alone cannot enforce
politeness because it doesn't know which coroutines share a host.

**The "no concurrency, ever" alternative.** Sequential rescan. Predictable,
zero coordination cost. But the wall-clock floor is `Σ (crawl + N · LLM +
DB)` per article — *most* of which is LLM latency that has nothing to do
with TechCrunch. We were leaving 3-4× wall-clock on the table.

**Why this split is correct.** The two concerns — "how busy is our
process" and "how busy is one upstream host" — are different. They have
different limits, set by different parties, varying over different time
scales. Modelling them as separate primitives means each can move
independently.

**What we give up.** Multi-host parallelism isn't exploited yet — with
only TechCrunch registered, the lock fully serialises all scraping. When
more crawlers are added the math improves automatically (different hosts
hold different locks).

We also accept a **per-host bottleneck**: even at `max_parallel = 8`,
each fetch to one host still happens one-at-a-time. Real speedup comes
from the LLM + DB overlap, not from parallel scraping. This is the
correct tradeoff — the rate limit lives at the host, not at our process.

---

## 6. Provenance as a first-class table

**Choice.** Every `Relationship` row is connected to one or more
`Provenance` rows, each pointing at the `Article` and quoting the
verbatim sentence that justified the edge.

**Alternative.** Store the quote as a column on `Relationship` itself.
Simpler schema, fewer joins.

**Why we didn't.** The same relationship can be supported by multiple
articles. "Elon Musk criticizes Sam Altman" appears in dozens of
TechCrunch articles over months — each one is independent evidence. A
one-quote-per-edge model has to choose: keep the first quote (lose
later context), keep the latest (lose history), or store an array
(reinvent provenance). Modelling provenance as its own table makes
"how do we know this is true?" a queryable property, which directly
answers the most important question a user can ask of an LLM-extracted
graph.

It also makes **the body-hash invalidation** clean: when an article's
content changes, we know exactly which `Provenance` rows to wipe
(`WHERE article_id = ?`), and the resulting orphan-detection pass
("relationships with no remaining provenance") catches the edges that
no longer have evidence and removes them.

**What we give up.** One extra join on the read path, one extra table on
the schema. The read API already serves provenance with the relationship
in a single query (eager-loaded), so the practical cost is negligible.

---

## 7. `BaseLLMClient` abstraction with two implementations

**Choice.** A one-method abstract base class — `structured_complete` — with
`OpenAIClient` (Responses API + `text_format`) and `LocalModelClient`
(any OpenAI-compatible HTTP server, schema-in-prompt + JSON-block parse)
as the two concrete implementations.

**Alternative.** Hardcode `openai.AsyncOpenAI` inside `LLMExtractor`.
Smaller surface, fewer files.

**Why we didn't.**

- **Cost / data-residency.** Some deployments cannot send article bodies
  to OpenAI (legal, residency, vendor lock-in concerns). The
  `LocalModelClient` makes those deployments possible without changing
  any business logic.
- **Cost-sensitive evaluation.** A nightly eval that runs the gold set
  against a local Llama model + an OpenAI model is a two-line config
  difference. The extractor doesn't know or care which is which.
- **Testing.** Stub `BaseLLMClient` is trivial; stub `openai.AsyncOpenAI`
  is not (the SDK's surface is large, the Responses API is opinionated).
  Resolver and chunk-checkpoint tests both lean on stub clients.

**What we give up.** Two API surfaces that diverge subtly — the Responses
API enforces JSON server-side; the local path has to parse a JSON block
out of free-text output. The `LocalModelClient`'s `_JSON_BLOCK_RE` is a
load-bearing regex that has to be conservative enough to recover from
markdown fences but lenient enough to skip unrelated chatter. We accept
that cost for the deployment flexibility.

---

## 8. Belt-and-suspenders org / placeholder filter

**Choice.** A small predicate `is_likely_organization(name)` plus
`filter_extraction(people, rels)` that drops org-shaped people, the
relationships that touch them, and placeholder markers ("Unknown",
"Anonymous", "staff writer"). Runs after every extraction.

**Why even with a tight prompt.** The extraction prompt explicitly tells
the LLM "no organizations as people, no companies, no products." That
prompt *mostly* works. The filter exists for the ~5% of chunks where the
model anthropomorphises an org anyway ("OpenAI announced …" → person
"OpenAI") or where the crawler couldn't find a byline and the prompt
fed it `Author: Unknown` (early bug — the model dutifully extracted
"Unknown" as a journalist person, then the resolver tried to merge every
future "Unknown" into one).

**Why the placeholder list is part of the same filter.** The two failure
modes are symmetric: "names that should not be in the people set."
Splitting them into two predicates would double the maintenance
surface for no benefit.

**What we give up.** A small false-positive risk — a real person named
"Sam Inc" (an LLC, a band, etc.) gets dropped by the suffix check. The
blocklist + suffix list are in one file each ([`app/extractors/filters.py`](./app/extractors/filters.py)),
editable in one line.

---

## 9. Evaluation harness with strict + fuzzy scoring

**Choice.** A small `gold/` set of hand-labelled fixture articles +
alias-resolution pairs. Score predicted people with set-based P/R/F1 on
normalised names; score predicted edges with **two** matching rules
side-by-side: strict (predicted `relation_type` ∈ gold `type_keywords`)
and fuzzy (any keyword is a substring of `relation_type`). Report both
per-article and aggregate.

**Why two scoring rules.** The extractor prompt is **open-vocabulary**
for `relation_type` — the model is free to say `criticizes`,
`condemns`, `slams`, `attacks`, etc. Locking the prompt to a closed
enum would make scoring trivial but throw away expressiveness; the same
relationship would be force-collapsed into a tag word that doesn't
match the article's tone.

The two rules bracket the truth:

- **Strict** is the pessimistic floor — it punishes synonym choice.
- **Fuzzy** is the synonym-tolerant ceiling — any one gold keyword as
  substring is enough.

The gap between them measures how much the model gets the *relationship*
right but uses different *words*. The actual quality is somewhere
between, and watching the gap close or widen tells you whether a prompt
change improved meaning or just vocabulary.

**Why we score resolver stages separately.** Two failure modes look
identical at the name layer but mean very different things: an LLM
fallback masking a Levenshtein bug is invisible to a pure-accuracy
metric. The stage-accuracy column tells you *which tier fired* for each
case, so a regression that pushes work from Stage 2 to Stage 3 shows up
even when total accuracy is unchanged.

---

## 10. Idempotent column migrations instead of Alembic

**Choice.** A `_migrate_pending_columns` hook ([`app/db/session.py`](./app/db/session.py))
runs after `create_all` on boot. For each column in a small known list,
it checks SQLAlchemy's inspector for presence and `ALTER TABLE ADD
COLUMN` if missing. Idempotent — no-op on fresh DBs (the column already
exists from `create_all`), no-op on up-to-date DBs (the column already
exists from a previous migration).

**Why not Alembic.** Alembic is the right answer for a project with
multiple long-lived database deployments and a team coordinating
schema versions. For this scope — one app, one schema we extended
exactly once (the chunk-checkpoint columns) — it would add:

- A separate revision tree and migration scripts to maintain.
- A CI step to keep autogenerated revisions in sync with model changes.
- An ops requirement to run `alembic upgrade head` on deploy.

That cost is real; the benefit kicks in at scales we're not at.

**Why not raw `CREATE TABLE` only.** Then any existing deployment (we
had one) loses data on schema change, because `create_all` doesn't add
columns to existing tables.

**What we give up.** Anything more complex than "add nullable column"
needs a real migration — schema drops, type changes, data backfills.
At that point Alembic earns its place. The current hook is explicitly
scoped to the additive case.

---

## 11. Two configurable resolver knobs

**Choice.** Two boolean settings,
`RESOLVER_RECENCY_ENABLED` and `RESOLVER_LLM_FALLBACK_ENABLED`, each
independently togglable. The four combinations cover the relevant
deployment profiles:

| recency | llm_fallback | profile |
|:---:|:---:|---|
| on | on | **Default** — max accuracy, accepts LLM cost on fuzzy matches. |
| on | off | **Cheap-with-coverage** — recency catches the easy mid-article cases; everything else becomes a fresh `Person` row, recoverable via batch dedupe. |
| off | on | **Strict + thorough** — refuse-on-ambiguity, LLM resolves everything fuzzy. Trades some merge accuracy for predictability. |
| off | off | **Fully deterministic** — zero LLM resolver calls, most duplicates. Cleanest semantics for environments that need to audit every decision. |

**Why these two knobs and not one.** They control orthogonal things:

- `recency` controls the *behaviour* of the resolver on ambiguous
  matches (guess from context vs refuse).
- `llm_fallback` controls *what kind* of call is allowed (LLM vs rules
  only).

Conflating them into a single "cost mode" toggle would force two
different deployment shapes (cheap-with-coverage vs strict+thorough) to
share the same flag, even though one wants fewer LLM calls and the
other wants more correctness.

**Why on by default.** The default is the highest-quality combination.
Cost-sensitive deployments opt out explicitly; the default does the
right thing for the median user.

---

## Appendix: when to revisit these choices

| Choice | Revisit when … |
|---|---|
| Per-chunk checkpoints | Average article shrinks to ≤ 2 chunks (the math stops mattering). |
| Five-stage resolver | Alias table grows past ~100k rows (Stage 2 Levenshtein becomes the bottleneck — add a trigram pre-filter or rapidfuzz). |
| Per-article recency | Cross-article patterns become useful (e.g. follow-on coverage in a multi-article story). Worth measuring before adding cross-article state. |
| Semaphore + per-host lock | More than ~10 crawlers; the lock-per-instance model is fine until you want global rate limiting across hosts. |
| Idempotent migration hook | Any schema change beyond "add nullable column." That's the trigger to adopt Alembic. |
| LLM client abstraction | A non-OpenAI-compatible provider (Anthropic's native API, etc.) comes online — add a third concrete class, the abstraction is already there. |
