# FE2C: Formula E to Consumer Optimization Engine

A single-file data pipeline plus agentic RAG system that treats Formula E as a lab for consumer EV battery work, with Lucid Motors (a Formula E supplier and a consumer EV maker) as the case study connecting the two.

## The problem

**Which Formula E battery-management strategy (energy density or regenerative-braking recovery) has the clearest transfer potential to consumer EV range, and how far can that transfer be quantified** when neither Formula E telemetry nor manufacturer regen data is published?

## The honest answer so far

It can't be quantified with public data, and this project is upfront about that. Formula E publishes no lap-level energy or regen data, so **the race stints are simulated** (seeded, `data_source = 'simulated_seeded'`), and the CA DMV feed returns a 2018 snapshot that predates Lucid's deliveries. What the project delivers is the **system**: a star schema, 12 data quality checks, a RAG store of sourced documentation, and an 8-tool Claude agent, built so that real stint data can replace the simulated layer without downstream changes.

On the current (simulated, no-team-effect) data the pipeline correctly finds **nothing**:

- Manufacturer efficiency ranking: Kruskal-Wallis p = 0.53. The order is noise.
- Range transfer for the focus team (Mahindra) vs. the field: −12.8% regen index, simulated Lucid Air range 402 mi (95% CI 389–416 mi around the 410 mi baseline), p = 0.29.

## Architecture: three layers, run in order

1. **Data pipeline (ETL).** Builds a SQLite star schema from simulated race stints, Open-Meteo weather and California DMV EV registrations (synthetic fallback when the download fails). Computes `regen_opportunity_index`, `fe2c_efficiency_score` (70% EER proxy + 30% velocity, normalised within battery generation) and `delta_e` (energy-cost proxy vs. race average).
2. **RAG vector store.** Chunks and embeds the project's technical write-up (`data/rag_docs/`) plus Wikipedia reference articles into ChromaDB.
3. **Agentic RAG (Claude).** Eight tools: SQL queries, semantic search, efficiency comparison, range simulation, cold-start circuit prediction, regen index ranking, CA market data and strategy recommendation, with prompt caching on the system prompt and tool schemas.

```bash
pip install -r requirements.txt
python3 fe2c.py --reset --skip-weather   # build the SQLite DB (drop --skip-weather to call Open-Meteo)
python3 fe2c.py --ingest                 # build the ChromaDB vector store (needs network for Wikipedia)
export ANTHROPIC_API_KEY=sk-ant-...
python3 fe2c.py --demo                   # 5 built-in showcase questions
python -m pytest tests/                  # offline: no network, no API key
```

## Reference facts (checked 2026-10-21)

- **Lucid in Formula E:** its Atieva division built the spec battery for **every** Gen2 car (2018–19 to 2021–22), and since 2023 Lucid supplies the Gen3 front drive unit, again to every team. It was never a Mahindra-specific supplier, so there's no "Lucid vs. non-Lucid" team comparison inside Formula E; Mahindra is tracked as a focus team only.
- **Lucid Air Sapphire:** EPA range 427 mi (687 km). The Air Grand Touring (516 mi) is the range leader.
- **Gen3 regen:** the FIA says over 40% of the energy used in a race comes from regenerative braking.

## Tools

Python, SQLite, ChromaDB, sentence-transformers, Anthropic Claude API (agentic tool use + RAG), pytest.

## Audit log

**2026-10-21 audit and rebuild**
- **Wrong reference facts, also baked into the RAG documents the agent cites as ground truth:** the Sapphire's range was 687 "miles" (it's 687 km / 427 mi, labelled "range leader"); Lucid was described as "powertrain supplier to the Mahindra team" (it supplied every team); the Gen3 recovery ratio was 22% (the FIA says over 40%); field energy draw was 1.8 kWh/km (impossible on a 54 kWh pack over ~95 km; ~0.57 kWh/km is the upper bound). All corrected in `fe2c.py` and `data/rag_docs/`.
- **The efficiency score was inverted.** `race_distance / eer_proxy` simplifies to avg lap × lap-time variance, an energy *cost*, so slow, erratic stints scored as most efficient. It now uses the EER proxy directly, and a test plants a more efficient team and checks it ranks first.
- **The range CI couldn't show a loss.** Both copies of the range simulation bootstrapped only the field and clipped the transfer factor at 0, so the "95% CI" (415.7–428.8 mi) excluded the baseline even at p = 0.25. One shared `range_transfer()` now resamples both groups and allows negative transfer.
- **The simulation wasn't internally consistent.** Drivers in the same race had different lap counts (the winner could do 39 laps, P11 48), finishing order was shuffled independently of pace, the same team always started on pole, DNFs could be classified ahead of finishers, and every race was dated a year early (before its own season, so the weather lookup used the wrong year). Now: one race distance per race, grid from qualifying pace, finish by laps then race time, and correct calendar years. Two quality checks were added for this (12 total).
- Added a Kruskal-Wallis test to the manufacturer ranking, and relabelled "Lucid chain" as "focus team".
- `formula_e_results.csv` was a stale export labelled `wiki_documented_telemetry_proxy`; it's regenerated from the pipeline and labelled `simulated_seeded`.
- Tests added (`tests/test_fe2c.py`, 13 tests, fully offline): pipeline build with all checks passing, race consistency, metric direction on planted data, range CI with no/negative/positive effects, the corrected facts, and seven of the eight agent tools against the built DB. (`search_race_documents` needs the ChromaDB store and network.)
- **Not tested here:** `--ingest` (Wikipedia is blocked from this environment) and the Claude chat loop (`--chat`/`--ask`/`--demo`, no API key).

**2026-09-16**
- `fetch_wikipedia_text()`: fixed the Gen3 article title (`Formula_E_Gen3`) and added pacing plus backoff on HTTP 429.
- Superseded model versions, redundant zips and a local settings file with absolute paths were removed. `db/` and generated outputs are gitignored.
