# Formula E is not a race series.
### It is a live laboratory for battery R&D.

> **[Live Report](https://jbattohokson.github.io/FormulaE_To_Consumer_Analysis/FE2C_Analysis.html)** | [GitHub Repo](https://github.com/jbattohokson/FormulaE_To_Consumer_Analysis)

---

## Executive Summary

An agentic RAG pipeline that models Formula E battery regen efficiency as a proxy for consumer EV range optimization. Formula E's hard energy cap per race makes every race a controlled experiment in battery management — teams that win by consuming less energy than rivals have demonstrated a transferable battery management insight. The FE2C (Formula E to Consumer) framework quantifies that transfer across three generations of Formula E battery technology, with Lucid Motors as the direct application case.

The pipeline orchestrates a SQLite star schema and ChromaDB vector store using a Claude tool-use agent with 8 custom tools and optimized prompt caching. The central analytical question for California EV buyers: does battery density (larger pack, longer range buffer) or recovery efficiency (better regen, lighter pack) produce more real-world range improvement?

| Metric | Value |
|--------|-------|
| Battery generations analyzed | Gen 1–3 (28 kWh to Gen 3 regen-focused architecture) |
| Lucid cell energy density | 300 Wh/kg — exceeds most market competitors |
| Gen 3 front axle regen | 250 kW — technical basis for regen efficiency analysis |
| Pipeline layers | 3 (SQLite star schema, ChromaDB vector store, Claude tool-use agent) |
| Custom agent tools | 8 (circuit analysis, efficiency scoring, CA market simulation, and more) |
| CA market data | California DMV EV registration data by brand, powertrain, ZIP code |

> **Project status:** Active research and engineering project. The pipeline, database schema, and agentic query layer are built and functional. Proxy metrics substituting for unavailable telemetry are documented explicitly throughout.

---

## Tools & Technologies

| Tool | Purpose |
|------|---------|
| Python | 7-stage run_pipeline() orchestration function |
| SQLite | Star schema — race_stints fact table + 4 dimension tables |
| dbt | Data quality layer — 10 targeted assertions, proxy assumptions documented |
| ChromaDB | Vector store — Formula E technical docs, FIA regulations, Lucid specs |
| Claude API (tool-use) | 8 custom tools, prompt caching, natural-language query interface |
| FastF1 / FIA API | Lap times, race results, gap data, pit events |
| Open-Meteo API | Ambient temperature, humidity, track temperature at stint start |

---

## Why Formula E Data Can Inform Consumer EV Decisions

Unlike combustion motorsports where fuel consumption is an efficiency target, Formula E teams face hard energy caps that function like a fixed battery charge. Exceeding allocated energy means slowing down or disqualification — a constraint that forces battery management optimization in a high-stakes environment where the cost of inefficiency is immediate and measurable.

| Generation | Battery Capability | Consumer EV Parallel |
|------------|-------------------|---------------------|
| Gen 1 (2014–2017) | ~28 kWh — teams swapped cars mid-race | Energy density baseline — proof of concept only |
| Gen 2 (2018–2020) | ~54 kWh — full race on single charge achieved | Range anxiety addressed for short-route EV use |
| Gen 3 (2023–present) | Charge speed, lighter packs, active thermal management | Aligns with consumer priorities: fast charging, range per kg |

Gen 3's design philosophy shift from energy density to recovery efficiency is a direct signal about where battery R&D ROI is moving. Consumer EV teams tracking this transition have a leading indicator for where cell engineering investment should be directed in the 2025–2028 window.

**California-specific answer:** Dense urban areas (LA Metro, Bay Area) favor recovery efficiency — more regen opportunities per mile and shorter trips where range buffer matters less. Rural and exurban areas (Central Valley, Inland Empire) favor density — fewer regen opportunities and larger distances between charging infrastructure make the range buffer more valuable. The FE2C simulation produces a ZIP-code-level recommendation map using CA DMV registration data weighted by these environmental factors.

Example agent queries the pipeline supports:
- "How does Lucid/Mahindra's regen index compare to the field in Gen3?"
- "What is the simulated range transfer from Gen3 race conditions to the Lucid Air EPA baseline?"
- "Which circuits favor regen-heavy strategies based on elevation and lap variance?"
- "What does the technical report say about the delta_e energy discipline signal?"

---

## Architecture: Three-layer agentic pipeline

### SQLite star schema
Fact table: race stints. Dimensions: battery hardware (Gen 1/2/3), track profile, manufacturer, environment. A dbt layer enforces data quality — no negative lap times, energy proxies within declared capacity bounds, environmental completeness checks.

### ChromaDB vector store
Formula E technical documentation, FIA regulations, and Lucid engineering publications embedded and indexed. Allows the agent to answer questions that require qualitative technical context alongside quantitative race data.

### Claude tool-use agent
8 custom tools covering circuit efficiency scoring, Delta-E calculation, regen opportunity indexing, cold-start circuit prediction, CA market simulation, and strategy recommendation by county. Answers multi-hop analytical questions against both structured data and semantic search simultaneously.

**Architecture rationale:** Formula E does not publish lap-level energy telemetry publicly. Proxy metrics are engineered from lap times, sector deltas, and gap data. The dbt quality layer documents every proxy assumption explicitly — the same standard production analytics environments use when perfect data does not exist.

---

## How Efficiency Is Measured in This Framework

### Delta-E Score — efficient battery management vs. costly position gains
Delta-E = Energy Consumed (proxy) vs. Positions Gained, calculated per stint using SQL window functions against the race average. A negative Delta-E indicates a team gained positions while consuming less energy than the stint average — the signature of efficient battery management. A negative Delta-E in a racing context maps to a real-world driver who gained range by driving more efficiently, not by having a larger pack.

### FE2C Efficiency Score — normalized cross-generation comparison
FE2C Efficiency Score = (Total Race Distance / Energy Used) × (Average Lap Velocity), normalized across generations using declared battery capacity differentials and adjusted for track profile using the aggression rating from the Track Profile dimension. Normalization across Gen 2 and Gen 3 requires assumptions about how capacity differentials map to efficiency comparability — those assumptions are documented in the dbt test layer and every comparison specifies which generation it applies to.

---

## Data Sources and Known Gaps

| Source | What It Provides | Status |
|--------|-----------------|--------|
| FIA / Formula E API | Lap times, race results, gap data, pit events | Live — ingested via pipeline |
| Open-Meteo API | Ambient temperature, humidity, track temperature | Live — weather enrichment layer |
| California DMV | EV registrations by brand, powertrain, ZIP code | Static — loaded at pipeline init |
| Lucid Motors / FIA documentation | Battery specs, Gen 3 regen specifications (250 kW front axle) | ChromaDB — semantic search layer |
| Formula E energy telemetry | Lap-level energy consumption | NOT AVAILABLE — proxy engineered from lap time and gap data |
| Raw regen efficiency data | Per-stint regen recovery amounts | NOT AVAILABLE — Regen Opportunity Index constructed from deceleration zone analysis |

---

## What This Analysis Cannot Tell You

### 1. The efficiency scores are proxy-based, not measured.
Formula E does not publish lap-level energy telemetry. Every efficiency metric in this project is derived from lap time deltas, gap data, and published battery capacity specifications. The proxy assumptions are documented in the dbt quality layer. FE2C Efficiency Scores should not be treated as equivalent to measured energy consumption data.

### 2. The Lucid-Mahindra technology transfer is assumed, not independently verified.
The project uses published reporting on Lucid's role as Mahindra's powertrain supplier as the basis for the technology transfer thesis. The specific engineering details of what transferred, when, and how are not publicly available. This is a plausible case study, not a confirmed causal chain.

### 3. The California simulation is a weighted estimate, not a predictive model.
The density vs. recovery recommendation by ZIP code is based on EV registration density and environmental proxies. It is a structured framework for thinking about the question — A/B testing or survey data from actual EV buyers in each region would be required to validate the recommendations.

In production analytics environments, the quality of a model is partly judged by how clearly its assumptions and limits are stated. Every proxy metric in this project is labeled as such in the pipeline output.
