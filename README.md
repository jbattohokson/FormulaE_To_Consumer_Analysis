# FE2C — Formula E to Consumer EV Pipeline

A 3-layer agentic RAG pipeline that treats Formula E racing as a live laboratory for consumer EV battery R&D. The system connects structured race telemetry to unstructured technical research and California DMV registration data, then makes the combined dataset queryable through a Claude-powered tool-use agent.

---

## What It Does

Formula E race stints generate regen efficiency signals — energy recovered under braking, per lap, per circuit — that are directly relevant to consumer EV range optimization. This project asks: how much of Lucid Motors' Gen3 regen advantage on track transfers to real-world range estimates, and does CA DMV registration data support that hypothesis on the demand side?

The agent can answer questions like:
- "How does Lucid/Mahindra's regen index compare to the field in Gen3?"
- "What is the simulated range transfer from Gen3 race conditions to the Lucid Air EPA baseline?"
- "Which circuits favor regen-heavy strategies based on elevation and lap variance?"
- "What does the technical report say about the delta_e energy discipline signal?"

---

## Architecture

The pipeline runs in three sequential layers. Each must complete before the next is useful.

```
Layer 1 — ETL / SQLite
  Builds a star-schema database from Formula E race data, Open-Meteo weather,
  and CA DMV EV registrations. Computes three proprietary efficiency metrics:
    - regen_opportunity_index: lap variance + elevation proxy for regen potential
    - fe2c_efficiency_score:   combined regen + energy discipline signal
    - delta_e:                 per-stint energy discipline (Technical Report §5.1)

Layer 2 — RAG / ChromaDB
  Chunks and embeds technical report excerpts, PDF files from data/rag_docs/,
  and Wikipedia articles on Formula E seasons, Mahindra Racing, Lucid Motors,
  and Gen3 car specs. Stored using sentence-transformer embeddings so the agent
  can cite project research rather than relying on general training knowledge.

Layer 3 — Agentic RAG / Anthropic Claude
  A tool-use agent with 8 custom tools spanning both layers. Uses prompt caching
  on the system prompt and tool schemas to reduce API cost across multi-step
  reasoning loops.
```

---

## Agent Tools

| Tool | What It Does |
|---|---|
| `query_race_database` | Executes SELECT queries against the SQLite star schema |
| `search_race_documents` | Semantic search over the ChromaDB vector store |
| `compare_manufacturer_efficiency` | Compares fe2c_efficiency_score across manufacturers |
| `simulate_consumer_range` | Models regen advantage transfer to Lucid Air EPA range |
| `cold_start_circuit_prediction` | Predicts regen index for circuits with no historical data |
| `calculate_regen_index` | Recomputes regen_opportunity_index from raw stint parameters |
| `get_ca_market_data` | Pulls CA DMV EV registration data filtered by brand or county |
| `enrich_rag_context` | Annotates race stints with retrieved qualitative context |

---

## Stack

- **Languages:** Python 3.13
- **Database:** SQLite (star schema), ChromaDB (vector store)
- **Agent:** Anthropic Claude API (tool use + prompt caching)
- **Embeddings:** sentence-transformers
- **Data:** pandas, NumPy, SciPy, requests
- **PDF parsing:** pypdf

---

## Setup & Execution

```bash
# Install dependencies
python3.13 -m pip install anthropic chromadb numpy pandas requests scipy \
  sentence-transformers pypdf

# Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# Layer 1: build SQLite database (required first)
python3.13 fe2c.py --reset

# Layer 2: build ChromaDB vector store
python3.13 fe2c.py --ingest

# Layer 3: interactive agent
python3.13 fe2c.py --chat

# Layer 3: single question
python3.13 fe2c.py --ask "How does Lucid's regen index compare to the field?"

# Layer 3: 5 built-in demo questions
python3.13 fe2c.py --demo
```

> **Python version:** Requires Python 3.13.x. chromadb and sentence-transformers do not ship binary wheels for Python 3.14+.

---

## Project Status

Active development. Core pipeline (Layers 1-3) is functional. The range simulation and cold-start prediction tools are operational. CA DMV integration and RAG enrichment are complete. README and documentation in progress.

---

## Notes on the Metrics

`regen_opportunity_index` is a proxy derived from lap time variance and circuit elevation — not live telemetry. The 15% regen weight applied in range simulations and the transfer factor are modeled assumptions, not calibrated parameters. All range estimates should be treated as directional. See Technical Report §6.2 and §8 for methodology.
