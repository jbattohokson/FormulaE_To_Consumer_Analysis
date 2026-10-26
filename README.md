# FE2C — Formula E to Consumer Optimization Engine

A single-file data engineering pipeline plus agentic RAG system that treats Formula E racing as a live laboratory for consumer EV battery R&D, using Lucid Motors (Formula E powertrain supplier and consumer EV maker) as the case study connecting the two.

**Business question:** which Formula E battery-management strategies (energy density vs. regenerative-braking recovery) show the clearest transfer potential to consumer EV range and performance, and how far can that transfer be quantified given that neither Formula E telemetry nor manufacturer regen data is publicly published?

## Architecture — three layers, run in order

1. **Data pipeline (ETL)**: builds a SQLite star schema from simulated Formula E race stints (640 stints over 5 seasons, generated with a fixed seed so every run matches), Open-Meteo weather, and live California DMV EV registration data. Computes three original efficiency metrics (`regen_opportunity_index`, `fe2c_efficiency_score`, `delta_e`).
2. **RAG vector store** — chunks and embeds the project's own technical write-up plus sourced Wikipedia reference material into ChromaDB, so the agent cites the project's actual research instead of general model knowledge.
3. **Agentic RAG (Claude)** — an 8-tool agent (SQLite queries, semantic search, efficiency comparisons, range simulation, cold-start prediction, regen index ranking, CA market data, strategy recommendation) with prompt caching on the system prompt and tool schemas.

```bash
pip install -r requirements.txt
python3 fe2c.py --reset      # build the SQLite DB
python3 fe2c.py --ingest     # build the ChromaDB vector store
export ANTHROPIC_API_KEY=sk-ant-...
python3 fe2c.py --demo       # 5 built-in showcase questions
```

## Tools

Python, SQLite, ChromaDB, sentence-transformers, Anthropic Claude API (agentic tool use + RAG).

## Data honesty — lead with this, don't bury it

- Formula E does not publish lap-level energy telemetry or regen efficiency data. Both are engineered proxies built from lap times, sector deltas, and published FIA Gen 2/Gen 3 specifications, and are labeled `[PROXY ESTIMATE]` in agent output — not presented as measured data. The pipeline itself prints this caveat (an "ADVERSARIAL NOTE") after every run of `--reset`, unprompted.
- The CA DMV open dataset returns a 2018 snapshot, which predates Lucid's Q4 2021 consumer deliveries. A calibrated synthetic supplement is used for Lucid registration counts, logged transparently in the pipeline.
- Race-level data is labeled as a documented telemetry proxy rather than presented as directly-measured race telemetry.

None of this is a weakness to hide. Leading with "every proxy is labeled, sourced, and distinguished from measured data" in interviews is exactly the kind of sentence that reads as analytical maturity rather than a project with unadmitted gaps.

## Testing status (2026-09-16)

Every layer was actually run against real data — 3x each for reliability — and two real bugs were found and fixed, not fabricated as passing:

- **Layer 1 (`--reset`)** — 3/3 clean, deterministic runs (identical FE2C scores every time: Lucid chain 23.20, field 26.51).
- **Layer 2 (`--ingest`)** — fixed two real issues in `fetch_wikipedia_text()`: (1) the Gen3 car article was fetched from `Gen3_(Formula_E)`, which 404s — the real title is `Formula_E_Gen3`, no parentheses; (2) firing 8 unpaced Wikipedia requests back-to-back reliably triggered 429 rate-limiting (3 of 8 articles failing in the worst run before the fix). Added request pacing and a backoff-and-retry specifically on 429. After the fix: 2/3 ingest runs completed with zero failures, 1/3 had a single transient 429 on one article (down from 3+ failures per run beforehand) — Wikipedia's public, unauthenticated rate limit is an external dependency this project can reduce the odds of hitting, not eliminate outright.
- **Layer 3 (the 8 agent tool functions)** — all 8 (`calculate_regen_index`, `query_race_database`, `search_race_documents`, `get_ca_market_data`, `compute_efficiency_comparison`, `get_range_simulation`, `predict_new_circuit_efficiency`, `get_strategy_recommendation`) were called directly against the real built database and vector store, 3x each: 24/24 calls succeeded with no code changes needed.
- **Not testable here:** the interactive Claude chat loop itself (`--chat` / `--ask` / `--demo`) needs `ANTHROPIC_API_KEY`, which isn't available in this environment. The eight underlying tool functions it calls were fully exercised instead, which is the part of the system with actual logic to break.

## Repo hygiene

Cleaned up 2026-09-16: `Model_Versions/` (twelve superseded iteration files), `Zip Folders/` (redundant zips of folders already present unzipped), and `.claude/settings.local.json` (exposed local absolute file paths) have been removed from this copy of the repo. `.gitignore` now excludes `db/fe2c.db`, `db/chroma/`, and the generated `data/week3_outputs/*.csv` — all regenerated by `--reset` / `--ingest`, so nothing is lost by not committing them.
