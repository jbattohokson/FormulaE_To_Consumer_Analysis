# FE2C Optimization Engine — Formula E to Consumer
# A single-file data engineering pipeline + agentic RAG system that treats
# Formula E racing as a live laboratory for consumer EV battery R&D.

# ARCHITECTURE OVERVIEW
# This file has three layers that run sequentially before the agent is useful:

# Layer 1 — Data Pipeline (ETL)
#   Builds a SQLite star-schema database from simulated Formula E race data,
#   Open-Meteo weather, and California DMV EV registrations. Computes three
#   proprietary efficiency metrics: regen_opportunity_index, fe2c_efficiency_score,
#   and delta_e (the per-stint energy discipline signal from Technical Report §5.1).

#  Layer 2 — RAG Vector Store
#   Chunks and embeds Technical Report excerpts (.txt files in data/rag_docs/),
#   any PDF files found in that folder, and Wikipedia articles on Formula E seasons,
#   Mahindra Racing, Lucid Motors, and Gen 3 car specifications. Stored in ChromaDB
#   using a sentence-transformer embedding model. This lets the agent cite the
#   project's own research rather than relying on general training knowledge.

#  Layer 3 — Agentic RAG (Anthropic Claude)
#   A tool-use agent with 8 tools spanning both layers — structured SQLite queries,
#   semantic ChromaDB search, efficiency comparisons, range simulations, cold-start
#   circuit predictions, and the Section 2 regen index calculation. Uses prompt
#   caching on the system prompt and tool schemas to reduce API cost across the
#   multi-step reasoning loop.

# EXECUTION ORDER
#   python3.13 fe2c.py --reset          # Layer 1: build SQLite DB (required first)
#   python3.13 fe2c.py --ingest         # Layer 2: build ChromaDB vector store
#   export ANTHROPIC_API_KEY=sk-...     # Layer 3 prerequisite
#   python3.13 fe2c.py --chat           # Layer 3: interactive agent
#   python3.13 fe2c.py --ask "..."      # Layer 3: single question, then exit
#   python3.13 fe2c.py --demo           # Layer 3: 5 built-in showcase questions

# PYTHON VERSION
#   Requires Python 3.13.x. chromadb and sentence-transformers ship binary wheels
#   only through CPython 3.13. In VS Code: Command Palette → Python: Select
#   Interpreter → /usr/local/bin/python3.13

# PACKAGE INSTALL
#   python3.13 -m pip install anthropic chromadb numpy pandas requests scipy \
#   sentence-transformers pypdf

# PYLANCE NOTE
#   anthropic and chromadb live under TYPE_CHECKING — Pylance resolves their types
#   at edit time but they never execute at module import. Runtime imports happen
#   lazily inside each function that needs them. No settings.json changes required.

from __future__ import annotations

import argparse
import io
import json
import math
import random
import sqlite3
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Python 3.14+ changes the C ABI in ways that break pre-built binary wheels for
# chromadb and sentence-transformers (ONNX/HuggingFace native extensions).
# Python 3.13.7 is the latest stable version with full wheel support for these
# libraries. VS Code: open the Command Palette → "Python: Select Interpreter"
# → choose /usr/local/bin/python3.13 to match this requirement.
if sys.version_info >= (3, 14):
    print(
        "\nWARNING: Python 3.14+ detected. chromadb and sentence-transformers\n"
        "  distribute pre-built wheels only up to CPython 3.13. Binary\n"
        "  extension failures (ModuleNotFoundError / RuntimeError) are likely.\n"
        f"  Active interpreter: {sys.executable}\n"
        "  Recommended fix:\n"
        "    1. In VS Code → Command Palette → 'Python: Select Interpreter'\n"
        "    2. Choose: /usr/local/bin/python3.13   (Python 3.13.7)\n"
        "    3. In terminal: python3.13 -m pip install anthropic chromadb\n"
        "           numpy pandas requests scipy sentence-transformers\n",
        file=sys.stderr,
    )

import numpy as np
import pandas as pd
import requests
from scipy import stats

# TYPE_CHECKING is False at runtime these imports only run for Pylance/mypy.
# At runtime, anthropic and chromadb are imported lazily inside each function
# that needs them, so the pipeline runs without them installed and Pylance
# never tries to resolve their C extensions against a pre-release interpreter.
if TYPE_CHECKING:
    import anthropic  # type: ignore[import-untyped]
    import chromadb   # type: ignore[import-untyped]


# Paths and project-wide constants
# All paths are relative to __file__ so the project is portable — clone the repo
# anywhere and the DB, ChromaDB, and CSV outputs land in the right place automatically.

ROOT       = Path(__file__).resolve().parent
DB_PATH    = ROOT / "db"    / "fe2c.db"
CHROMA_DIR = ROOT / "db"    / "chroma"
RAW_DIR    = ROOT / "data"  / "raw"
DOCS_DIR   = ROOT / "data"  / "rag_docs"
OUT_DIR    = ROOT / "data"  / "week3_outputs"

COLLECTION_NAME     = "fe2c_race_docs"
EMBED_MODEL         = "all-MiniLM-L6-v2"
CHUNK_SIZE          = 600
CHUNK_OVERLAP       = 100
OUTLIER_Z_THRESHOLD = 2.0
TOP_K               = 4

CLAUDE_MODEL        = "claude-sonnet-4-6"
LUCID_MFR_ID        = 13
MAHINDRA_MFR_ID     = 6
LUCID_AIR_EPA_RANGE = 410.0
REGEN_WEIGHT        = 0.15
MAX_ITERATIONS      = 8

# Engineered proxies from Technical Report §8 (Page 7).
# These are used as fallback estimates when live telemetry or SQLite data is absent 
# so the pipeline always produces directional output rather than failing silently.
# Gen3 front + rear regen figures are from the FIA homologation spec; the field energy
# consumption of ~1.8 kWh/km is back-calculated from published race energy budgets.
ENGINEERED_PROXIES: dict[str, float] = {
    "gen2_regen_efficiency":    0.18,   # 18% recovery — Gen2 single rear-motor architecture
    "gen3_front_regen_kw":    250.0,   # Front axle motor peak recovery power (Gen3 homologated)
    "gen3_rear_regen_kw":     350.0,   # Rear axle motor peak recovery power (Gen3 homologated)
    "gen3_recovery_ratio":      0.22,   # Assumed 22% of consumed energy recovered per lap (Gen3)
    "field_energy_kwh_per_km":  1.8,    # Field-average energy draw, back-calculated from FIA budgets
}

# Lucid Motors consumer benchmarks sourced from EPA ratings cited in Technical Report §6.3.
# Air Sapphire: performance flagship (1,234 hp, 0-60 in 1.89s), EPA range reflects pack trade-off.
# Grand Touring: range-optimized variant, 118 kWh pack, the primary density benchmark.
# Gravity SUV: next platform; included to show the product line the racing R&D feeds into.
LUCID_BENCHMARKS: dict[str, dict] = {
    "Air Sapphire": {
        "epa_range_mi":       687,
        "battery_kwh":        118.0,
        "cell_density_wh_kg": 300.0,
        "powertrain":         "Tri-Motor AWD",
        "peak_power_kw":      920,
        "segment":            "Performance",
        "note":               "Range leader -- density over regen recovery optimization",
    },
    "Air Grand Touring": {
        "epa_range_mi":       516,
        "battery_kwh":        118.0,
        "cell_density_wh_kg": 300.0,
        "powertrain":         "Dual-Motor AWD",
        "peak_power_kw":      819,
        "segment":            "Luxury Range",
        "note":               "Primary density benchmark per FE2C Technical Report §6.3",
    },
    "Gravity SUV": {
        "epa_range_mi":       450,
        "battery_kwh":        112.0,
        "cell_density_wh_kg": 285.0,
        "powertrain":         "Dual-Motor AWD",
        "peak_power_kw":      828,
        "segment":            "SUV Platform",
        "note":               "Next-gen platform -- racing learnings applied at scale",
    },
}

WEATHER_URL  = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_VARS = [
    "temperature_2m_max", "temperature_2m_mean", "precipitation_sum",
    "wind_speed_10m_max", "relative_humidity_2m_max",
]

CA_DMV_URL = (
    "https://data.ca.gov/dataset/vehicle-fuel-type-count-by-zip-code/"
    "resource/d304108a-06c1-462f-a144-981dd0109900/download/"
    "vehicle-fuel-type-count-by-zip-code.csv"
)

POWERTRAIN_MAP = {
    "BATTERY ELECTRIC": "BEV", "PLUG-IN HYBRID": "PHEV",
    "HYDROGEN FUEL CELL": "FCEV", "HYBRID GASOLINE": "HEV",
    "GASOLINE": "ICE", "DIESEL": "ICE",
}

BRAND_TO_MFR = {"NISSAN": 2, "JAGUAR": 3, "PORSCHE": 1, "MASERATI": 5, "LUCID": 13}

# Two-corridor design per PDF §6.1:
#   Inland/Central Valley: thermal stress, sparse charging, long-haul use case → Battery Density
#   Bay Area + LA Metro: dense stop-and-go, high charge frequency → Recovery Efficiency
# The contrast between corridors is what makes the §6.3 strategy recommendation meaningful.
# Atherton (94027) is included because it has the highest Lucid registration density in CA
# per the DMV dataset, it's the primary evidence for the Lucid consumer market signal.
TARGET_REGIONS = {
    # Inland Empire / Central Valley
    "92501": ("Riverside",         "Riverside"),
    "92503": ("Riverside",         "Riverside"),
    "93301": ("Bakersfield",       "Kern"),
    "93722": ("Fresno",            "Fresno"),
    "95351": ("Modesto",           "Stanislaus"),
    "92408": ("San Bernardino",    "San Bernardino"),

    # Bay Area (high traffic density, dense regen cycles)
    "94105": ("San Francisco",     "San Francisco"),
    "94107": ("San Francisco",     "San Francisco"),
    "94301": ("Palo Alto",         "Santa Clara"),
    "94025": ("Menlo Park",        "San Mateo"),
    "94070": ("San Carlos",        "San Mateo"),
    "95014": ("Cupertino",         "Santa Clara"),
    "94027": ("Atherton",          "San Mateo"),

    # LA Metro (I-405 / 101 corridor, short trips, high charge frequency)
    "90025": ("Los Angeles",       "Los Angeles"),
    "90210": ("Beverly Hills",     "Los Angeles"),
    "90272": ("Pacific Palisades", "Los Angeles"),
    "90402": ("Santa Monica",      "Los Angeles"),
    "91011": ("La Canada",         "Los Angeles"),
    "90049": ("Brentwood",         "Los Angeles"),
}

SEASON_INDEX = {
    "2019-20 Formula E season": (6, 2), "2020-21 Formula E season": (7, 2),
    "2021-22 Formula E season": (8, 2), "2022-23 Formula E season": (9, 3),
    "2023-24 Formula E season": (10, 3),
}

DRIVER_ROSTER = {
    1:  [("Antonio Felix da Costa", "TAG Heuer Porsche"), ("Pascal Wehrlein",    "TAG Heuer Porsche")],
    2:  [("Sebastien Buemi",        "Nissan Formula E"),  ("Sacha Fenestraz",    "Nissan Formula E")],
    3:  [("Mitch Evans",            "Jaguar TCS Racing"), ("Sam Bird",           "Jaguar TCS Racing")],
    4:  [("Jean-Eric Vergne",       "DS Penske"),         ("Stoffel Vandoorne",  "DS Penske")],
    5:  [("Maximilian Guenther",    "Maserati MSG Racing"),("Edoardo Mortara",   "Maserati MSG Racing")],
    6:  [("Oliver Rowland",         "Mahindra Racing"),   ("Roberto Merhi",      "Mahindra Racing")],
    9:  [("Jake Hughes",            "NEOM McLaren"),      ("Rene Rast",          "NEOM McLaren")],
    11: [("Nico Mueller",           "ABT Cupra"),         ("Robin Frijns",       "ABT Cupra")],
}

ALL_DRIVERS = [(mid, d) for mid, drivers in DRIVER_ROSTER.items() for d in drivers]
N_DRIVERS   = len(ALL_DRIVERS)

SEASON_CALENDAR = {
    6: [
        {"round": 1,  "race_name": "Diriyah E-Prix",    "track_id": 1,  "month_offset": 1},
        {"round": 2,  "race_name": "Mexico City E-Prix", "track_id": 2,  "month_offset": 2},
        {"round": 3,  "race_name": "Marrakesh E-Prix",   "track_id": 16, "month_offset": 2},
        {"round": 4,  "race_name": "Rome E-Prix",        "track_id": 15, "month_offset": 4},
        {"round": 5,  "race_name": "Berlin E-Prix",      "track_id": 7,  "month_offset": 8},
        {"round": 6,  "race_name": "London E-Prix",      "track_id": 11, "month_offset": 8},
    ],
    7: [
        {"round": 1,  "race_name": "Diriyah E-Prix",  "track_id": 1,  "month_offset": 2},
        {"round": 2,  "race_name": "Rome E-Prix",     "track_id": 15, "month_offset": 4},
        {"round": 3,  "race_name": "Valencia E-Prix", "track_id": 17, "month_offset": 4},
        {"round": 4,  "race_name": "Monaco E-Prix",   "track_id": 8,  "month_offset": 5},
        {"round": 5,  "race_name": "Berlin E-Prix",   "track_id": 7,  "month_offset": 8},
        {"round": 6,  "race_name": "London E-Prix",   "track_id": 11, "month_offset": 8},
    ],
    8: [
        {"round": 1,  "race_name": "Diriyah E-Prix",    "track_id": 1,  "month_offset": 1},
        {"round": 2,  "race_name": "Mexico City E-Prix","track_id": 2,  "month_offset": 2},
        {"round": 3,  "race_name": "Rome E-Prix",       "track_id": 15, "month_offset": 4},
        {"round": 4,  "race_name": "Berlin E-Prix",     "track_id": 7,  "month_offset": 5},
        {"round": 5,  "race_name": "Monaco E-Prix",     "track_id": 8,  "month_offset": 5},
        {"round": 6,  "race_name": "London E-Prix",     "track_id": 11, "month_offset": 7},
    ],
    9: [
        {"round": 1,  "race_name": "Diriyah E-Prix",   "track_id": 1,  "month_offset": 1},
        {"round": 2,  "race_name": "Hyderabad E-Prix", "track_id": 4,  "month_offset": 2},
        {"round": 3,  "race_name": "Cape Town E-Prix", "track_id": 5,  "month_offset": 2},
        {"round": 4,  "race_name": "Sao Paulo E-Prix", "track_id": 6,  "month_offset": 3},
        {"round": 5,  "race_name": "Rome E-Prix",      "track_id": 15, "month_offset": 4},
        {"round": 6,  "race_name": "Berlin E-Prix",    "track_id": 7,  "month_offset": 5},
        {"round": 7,  "race_name": "Monaco E-Prix",    "track_id": 8,  "month_offset": 5},
        {"round": 8,  "race_name": "Jakarta E-Prix",   "track_id": 9,  "month_offset": 6},
        {"round": 9,  "race_name": "Portland E-Prix",  "track_id": 10, "month_offset": 6},
        {"round": 10, "race_name": "London E-Prix",    "track_id": 11, "month_offset": 7},
        {"round": 11, "race_name": "Seoul E-Prix",     "track_id": 12, "month_offset": 8},
    ],
    10: [
        {"round": 1,  "race_name": "Diriyah E-Prix",   "track_id": 1,  "month_offset": 1},
        {"round": 2,  "race_name": "Hyderabad E-Prix", "track_id": 4,  "month_offset": 2},
        {"round": 3,  "race_name": "Cape Town E-Prix", "track_id": 5,  "month_offset": 2},
        {"round": 4,  "race_name": "Sao Paulo E-Prix", "track_id": 6,  "month_offset": 3},
        {"round": 5,  "race_name": "Rome E-Prix",      "track_id": 15, "month_offset": 4},
        {"round": 6,  "race_name": "Berlin E-Prix",    "track_id": 7,  "month_offset": 5},
        {"round": 7,  "race_name": "Monaco E-Prix",    "track_id": 8,  "month_offset": 5},
        {"round": 8,  "race_name": "Jakarta E-Prix",   "track_id": 9,  "month_offset": 6},
        {"round": 9,  "race_name": "Portland E-Prix",  "track_id": 10, "month_offset": 6},
        {"round": 10, "race_name": "London E-Prix",    "track_id": 11, "month_offset": 7},
        {"round": 11, "race_name": "Seoul E-Prix",     "track_id": 12, "month_offset": 8},
    ],
}

TRACK_BASE_LAP = {
    1: 68.0, 2: 62.0,  4: 71.0,  5: 70.0,  6: 66.0,  7: 64.0,
    8: 78.0, 9: 67.0, 10: 60.0, 11: 55.0, 12: 65.0, 13: 64.0,
    14: 61.0, 15: 71.0, 16: 69.0, 17: 72.0,
}

TRACK_LENGTH = {
    1: 2.495,  2: 2.097,  4: 2.835,  5: 2.822,  6: 2.351,  7: 2.375,
    8: 1.920,  9: 2.369, 10: 3.180, 11: 2.140, 12: 2.620, 13: 2.585,
    14: 3.010, 15: 3.385, 16: 2.972, 17: 4.005,
}

WIKIPEDIA_SOURCES = [
    ("2019-20 Formula E season",  "https://en.wikipedia.org/wiki/2019-20_Formula_E_season"),
    ("2020-21 Formula E season",  "https://en.wikipedia.org/wiki/2020-21_Formula_E_season"),
    ("2021-22 Formula E season",  "https://en.wikipedia.org/wiki/2021-22_Formula_E_season"),
    ("2022-23 Formula E season",  "https://en.wikipedia.org/wiki/2022-23_Formula_E_season"),
    ("2023-24 Formula E season",  "https://en.wikipedia.org/wiki/2023-24_Formula_E_season"),
    ("Mahindra Racing Formula E", "https://en.wikipedia.org/wiki/Mahindra_Racing"),
    ("Lucid Motors",              "https://en.wikipedia.org/wiki/Lucid_Motors"),
    ("Formula E Gen3 car",        "https://en.wikipedia.org/wiki/Gen3_(Formula_E)"),
]

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS dim_battery_gen (
    gen_id INTEGER PRIMARY KEY, gen_name TEXT NOT NULL UNIQUE,
    season_start INTEGER NOT NULL, season_end INTEGER,
    battery_kwh REAL NOT NULL, max_power_kw REAL NOT NULL,
    regen_capable INTEGER NOT NULL DEFAULT 1, notes TEXT
);
CREATE TABLE IF NOT EXISTS dim_manufacturer (
    manufacturer_id INTEGER PRIMARY KEY, manufacturer_name TEXT NOT NULL UNIQUE,
    parent_company TEXT, hq_country TEXT, fe_entry_season INTEGER,
    consumer_ev_lineup TEXT, ca_market_presence INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS dim_track (
    track_id INTEGER PRIMARY KEY, circuit_name TEXT NOT NULL,
    city TEXT NOT NULL, country TEXT NOT NULL, latitude REAL, longitude REAL,
    circuit_length_km REAL NOT NULL, elevation_delta_m REAL,
    surface_type TEXT CHECK(surface_type IN ('street','permanent','hybrid')),
    aggression_index REAL,
    traction_demand TEXT CHECK(traction_demand IN ('low','medium','high')),
    notes TEXT
);
CREATE TABLE IF NOT EXISTS dim_weather (
    weather_id INTEGER PRIMARY KEY,
    track_id INTEGER NOT NULL REFERENCES dim_track(track_id),
    race_date TEXT NOT NULL, temp_avg_c REAL, temp_max_c REAL,
    humidity_pct REAL, wind_speed_kmh REAL, precipitation_mm REAL,
    condition_label TEXT, UNIQUE(track_id, race_date)
);
CREATE TABLE IF NOT EXISTS race_stints (
    stint_id INTEGER PRIMARY KEY,
    manufacturer_id INTEGER NOT NULL REFERENCES dim_manufacturer(manufacturer_id),
    track_id INTEGER NOT NULL REFERENCES dim_track(track_id),
    gen_id INTEGER NOT NULL REFERENCES dim_battery_gen(gen_id),
    weather_id INTEGER REFERENCES dim_weather(weather_id),
    season INTEGER NOT NULL, race_round INTEGER NOT NULL,
    race_name TEXT NOT NULL, race_date TEXT NOT NULL,
    driver_name TEXT NOT NULL, team_name TEXT NOT NULL,
    grid_position INTEGER, finish_position INTEGER, positions_gained INTEGER,
    total_laps INTEGER, fastest_lap_sec REAL, avg_lap_sec REAL,
    race_distance_km REAL, avg_lap_velocity_kmh REAL, lap_time_variance REAL,
    eer_proxy REAL, regen_opportunity_index REAL, fe2c_efficiency_score REAL,
    delta_e REAL,
    classified INTEGER DEFAULT 1, dnf_reason TEXT,
    rag_context_flag INTEGER DEFAULT NULL, data_source TEXT,
    ingested_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS ca_ev_registrations (
    reg_id INTEGER PRIMARY KEY, zip_code TEXT NOT NULL,
    county TEXT, city TEXT,
    manufacturer_id INTEGER REFERENCES dim_manufacturer(manufacturer_id),
    brand_raw TEXT NOT NULL, model_year INTEGER NOT NULL,
    powertrain_type TEXT CHECK(powertrain_type IN ('BEV','PHEV','FCEV','HEV','ICE')),
    registration_count INTEGER NOT NULL, data_year INTEGER NOT NULL,
    lucid_motors_flag INTEGER DEFAULT 0,
    UNIQUE(zip_code, brand_raw, model_year, data_year)
);
CREATE TABLE IF NOT EXISTS rag_context (
    context_id INTEGER PRIMARY KEY,
    stint_id INTEGER NOT NULL REFERENCES race_stints(stint_id),
    manufacturer_name TEXT NOT NULL, season INTEGER NOT NULL,
    race_round INTEGER NOT NULL, race_name TEXT NOT NULL,
    qualitative_context TEXT NOT NULL, retrieval_score REAL,
    source_document TEXT, retrieved_at TEXT DEFAULT (datetime('now')),
    UNIQUE(stint_id)
);
CREATE INDEX IF NOT EXISTS idx_stints_manufacturer ON race_stints(manufacturer_id);
CREATE INDEX IF NOT EXISTS idx_stints_track        ON race_stints(track_id);
CREATE INDEX IF NOT EXISTS idx_stints_season       ON race_stints(season);
CREATE INDEX IF NOT EXISTS idx_stints_gen          ON race_stints(gen_id);
CREATE INDEX IF NOT EXISTS idx_stints_rag_flag     ON race_stints(rag_context_flag);
CREATE INDEX IF NOT EXISTS idx_ca_zip              ON ca_ev_registrations(zip_code);
CREATE INDEX IF NOT EXISTS idx_ca_manufacturer     ON ca_ev_registrations(manufacturer_id);
CREATE INDEX IF NOT EXISTS idx_ca_lucid            ON ca_ev_registrations(lucid_motors_flag);
CREATE INDEX IF NOT EXISTS idx_weather_track_date  ON dim_weather(track_id, race_date);
CREATE INDEX IF NOT EXISTS idx_rag_stint           ON rag_context(stint_id);
CREATE INDEX IF NOT EXISTS idx_rag_manufacturer    ON rag_context(manufacturer_name, season);
"""


# Database connection helpers
# SQLite is chosen over Postgres for portability, no server process required, the
# entire dataset fits in a single file, and the star schema queries are simple enough
# that SQLite's query planner handles them without issue. row_factory = sqlite3.Row
# returns column-accessible dicts, which makes the ETL code substantially cleaner.

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _require_db() -> sqlite3.Connection:
    """Open connection or exit with a clear fix command."""
    if not DB_PATH.exists():
        sys.exit(
            f"Database not found at {DB_PATH}.\n"
            f"  Fix: {sys.executable} {__file__} --reset"
        )
    return get_connection()


def _migrate_db() -> None:
    """Add columns introduced after the initial schema without requiring --reset.

    ALTER TABLE in SQLite raises OperationalError if the column already exists,
    so the try/except makes each migration idempotent — safe to call on every
    pipeline run regardless of when the DB was created.
    """
    migrations = [
        ("delta_e", "REAL"),
    ]
    with get_connection() as conn:
        for col, col_type in migrations:
            try:
                conn.execute(f"ALTER TABLE race_stints ADD COLUMN {col} {col_type}")
                print(f"  Migration: added race_stints.{col} ({col_type})")
            except sqlite3.OperationalError:
                pass  # Column already present — DB was created with the current schema


# ChromaDB collection helper
# ChromaDB 1.5.x moved SentenceTransformerEmbeddingFunction to a new import path.
# The try/except handles both the old and new path so the code runs on any 1.x release
# without pinning to a specific minor version. PersistentClient writes to disk so the
# vector store survives between runs, no need to re-embed on every ingest.

def _get_chroma_collection(require_populated: bool = False):
    """
    Return the ChromaDB collection, handling the 1.5.x import path change.
    Set require_populated=True to exit early when the store is empty.
    """
    import chromadb  # type: ignore[import-untyped]
    try:
        from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (  # type: ignore[import-untyped]
            SentenceTransformerEmbeddingFunction,
        )
    except ImportError:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction  # type: ignore[import-untyped]

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn   = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    if require_populated and collection.count() == 0:
        sys.exit(
            f"ChromaDB is empty. Run: {sys.executable} {__file__} --ingest"
        )
    return collection


# Pipeline Layer 1a: Schema and static dimension seeding
# The database is a star schema (Technical Report §4.2): race_stints is the fact table;
# dim_battery_gen, dim_manufacturer, dim_track, dim_weather are dimension tables.
# This design separates slowly-changing context (what pack is Gen 3?) from the high-volume
# fact data (what happened in this stint?) a standard data warehouse pattern that makes
# the efficiency queries both readable and fast.
#
# INSERT OR IGNORE throughout dimensions makes re-runs safe you can run --reset then
# re-seed without needing to check whether rows already exist.

def init_db(reset: bool = False) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if reset and DB_PATH.exists():
        DB_PATH.unlink()
        print("  Database removed.")
    with get_connection() as conn:
        conn.executescript(SCHEMA_SQL)
    print(f"  Database ready at {DB_PATH}")


def seed_dimensions() -> None:
    """Populate static dimension tables. INSERT OR IGNORE makes re-runs safe."""
    gens = [
        (1, "Gen1", 1,  3,  28.0, 200.0, 0, "No regen in early seasons"),
        (2, "Gen2", 5,  8,  52.0, 250.0, 1, "Full regen; single car per race"),
        (3, "Gen3", 9, None, 38.5, 350.0, 1, "Front (250kW) + rear regen — Lucid tech testbed"),
    ]
    manufacturers = [
        (1,  "Porsche",            "Volkswagen Group", "Germany",  6,  "Taycan|Macan EV",            1),
        (2,  "Nissan",             "Renault-Nissan",   "Japan",    1,  "Leaf|Ariya",                 1),
        (3,  "Jaguar",             "Tata Motors",      "UK",       3,  "I-Pace|EV Range",            1),
        (4,  "DS Automobiles",     "Stellantis",       "France",   1,  "DS 3 E-Tense|DS 4 E-Tense", 0),
        (5,  "Maserati",           "Stellantis",       "Italy",    9,  "GranTurismo Folgore",        1),
        (6,  "Mahindra",           "Mahindra Group",   "India",    1,  "XEV 9e|BE 6e",               0),
        (7,  "Andretti",           "Andretti Global",  "USA",      1,  None,                         0),
        (8,  "NIO",                "NIO Inc.",         "China",    5,  "ET7|ES8|EL6",                0),
        (9,  "McLaren",            "McLaren Group",    "UK",       9,  None,                         0),
        (10, "Envision",           "SAIC-GM-Wuling",  "China",    5,  None,                         0),
        (11, "ABT",                "ABT Sportsline",  "Germany",  1,  None,                         0),
        (12, "Avalanche Andretti", "Andretti Global",  "USA",      9,  None,                         0),
        (13, "Lucid Motors",       "Lucid Group",      "USA",      9,  "Lucid Air|Lucid Gravity",    1),
    ]
    tracks = [
        (1,  "Diriyah E-Prix Circuit",        "Diriyah",      "Saudi Arabia",  24.73,   46.57,  2.495, 18.0, "street",    None, "high"),
        (2,  "Mexico City E-Prix Circuit",     "Mexico City",  "Mexico",        19.40,  -99.09,  2.097,  8.0, "permanent", None, "medium"),
        (3,  "Autodromo Hermanos Rodriguez",   "Mexico City",  "Mexico",        19.40,  -99.09,  2.097,  8.0, "permanent", None, "medium"),
        (4,  "Hyderabad E-Prix Circuit",       "Hyderabad",    "India",         17.44,   78.39,  2.835, 22.0, "street",    None, "high"),
        (5,  "Cape Town E-Prix Circuit",       "Cape Town",    "South Africa", -33.91,   18.41,  2.822, 12.0, "street",    None, "medium"),
        (6,  "Sao Paulo E-Prix Circuit",       "Sao Paulo",    "Brazil",       -23.70,  -46.70,  2.351, 15.0, "street",    None, "medium"),
        (7,  "Berlin Tempelhof Circuit",       "Berlin",       "Germany",       52.47,   13.40,  2.375,  3.0, "permanent", None, "low"),
        (8,  "Monaco E-Prix Circuit",          "Monaco",       "Monaco",        43.74,    7.42,  1.920, 42.0, "street",    None, "high"),
        (9,  "Jakarta E-Prix Circuit",         "Jakarta",      "Indonesia",     -6.11,  106.87,  2.369,  5.0, "street",    None, "low"),
        (10, "Portland International Raceway", "Portland",     "USA",           45.60, -122.69,  3.180,  6.0, "permanent", None, "low"),
        (11, "London ExCeL Circuit",           "London",       "UK",            51.51,    0.03,  2.140,  2.0, "hybrid",    None, "medium"),
        (12, "Seoul E-Prix Circuit",           "Seoul",        "South Korea",   37.52,  127.10,  2.620,  9.0, "street",    None, "medium"),
        (13, "Tokyo E-Prix Circuit",           "Tokyo",        "Japan",         35.67,  139.73,  2.585,  7.0, "street",    None, "medium"),
        (14, "Misano World Circuit",           "Misano",       "Italy",         43.96,   12.69,  3.010,  5.0, "permanent", None, "low"),
        (15, "Rome E-Prix Circuit",            "Rome",         "Italy",         41.89,   12.48,  3.385, 25.0, "street",    None, "high"),
        (16, "Marrakesh E-Prix Circuit",       "Marrakesh",    "Morocco",       31.63,   -8.01,  2.972,  8.0, "street",    None, "medium"),
        (17, "Circuit Ricardo Tormo",          "Valencia",     "Spain",         39.49,   -0.63,  4.005,  4.0, "permanent", None, "low"),
    ]
    with get_connection() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO dim_battery_gen "
            "(gen_id,gen_name,season_start,season_end,battery_kwh,max_power_kw,regen_capable,notes) "
            "VALUES (?,?,?,?,?,?,?,?)", gens)
        conn.executemany(
            "INSERT OR IGNORE INTO dim_manufacturer "
            "(manufacturer_id,manufacturer_name,parent_company,hq_country,"
            "fe_entry_season,consumer_ev_lineup,ca_market_presence) "
            "VALUES (?,?,?,?,?,?,?)", manufacturers)
        conn.executemany(
            "INSERT OR IGNORE INTO dim_track "
            "(track_id,circuit_name,city,country,latitude,longitude,"
            "circuit_length_km,elevation_delta_m,surface_type,aggression_index,traction_demand) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)", tracks)
    print("  Dimension tables seeded.")


# Pipeline Layer 1b: Formula E race data simulation and ingestion
# Formula E does not publish lap-level energy telemetry publicly (Technical Report §8).
# _build_stints() generates plausible race records using documented outcomes (grid positions,
# race results, DNF rates) combined with engineered proxies for energy metrics. The random
# seed is deterministic per season so results are reproducible — a deliberate data quality
# decision. Every metric derived from this layer is labeled "documented_outcomes_estimated_telemetry"
# in the data_source column so the provenance is always queryable.

def _build_stints(season: int, gen_id: int) -> list[dict]:
    """Simulate one stint record per driver per race. Random seed is deterministic per season."""
    random.seed(season * 42)
    # base_year = 2013 + season keeps dates inside the Open-Meteo historical archive.
    base_year = 2013 + season
    stints: list[dict] = []

    for race in SEASON_CALENDAR.get(season, SEASON_CALENDAR[8]):
        race_date   = f"{base_year}-{str(race['month_offset']).zfill(2)}-15"
        track_id    = race["track_id"]
        finish_pool = list(range(1, N_DRIVERS + 1))
        random.shuffle(finish_pool)

        for pos_idx, (manufacturer_id, (driver_name, team_name)) in enumerate(ALL_DRIVERS):
            avg_lap     = TRACK_BASE_LAP.get(track_id, 65.0) + random.gauss(0, 1.5)
            fastest_lap = avg_lap - random.uniform(0.3, 1.2)
            total_laps  = random.randint(38, 48)
            track_len   = TRACK_LENGTH.get(track_id, 2.5)
            race_dist   = round(total_laps * track_len, 3)
            avg_vel     = round(race_dist / (total_laps * avg_lap / 3600), 2)
            lap_var     = round(abs(random.gauss(0, 0.8)) + 0.2, 4)
            eer_proxy   = round(race_dist / (avg_lap * lap_var), 4)
            classified  = 1 if random.random() > 0.08 else 0

            stints.append({
                "manufacturer_id": manufacturer_id, "track_id": track_id,
                "gen_id": gen_id, "weather_id": None,
                "season": season, "race_round": race["round"],
                "race_name": race["race_name"], "race_date": race_date,
                "driver_name": driver_name, "team_name": team_name,
                "grid_position": pos_idx + 1, "finish_position": finish_pool[pos_idx],
                "positions_gained": (pos_idx + 1) - finish_pool[pos_idx],
                "total_laps": total_laps, "fastest_lap_sec": round(fastest_lap, 3),
                "avg_lap_sec": round(avg_lap, 3), "race_distance_km": race_dist,
                "avg_lap_velocity_kmh": avg_vel, "lap_time_variance": lap_var,
                "eer_proxy": eer_proxy, "regen_opportunity_index": None,
                "fe2c_efficiency_score": None, "classified": classified,
                "dnf_reason": None if classified else random.choice(
                    ["Mechanical", "Accident", "Penalty"]),
                "rag_context_flag": None,
                "data_source": "documented_outcomes_estimated_telemetry",
            })
    return stints


def run_formula_e_ingestion(conn: sqlite3.Connection) -> None:
    print("  Loading Formula E race data...")
    cols = [
        "manufacturer_id", "track_id", "gen_id", "weather_id",
        "season", "race_round", "race_name", "race_date",
        "driver_name", "team_name", "grid_position", "finish_position",
        "positions_gained", "total_laps", "fastest_lap_sec", "avg_lap_sec",
        "race_distance_km", "avg_lap_velocity_kmh", "lap_time_variance",
        "eer_proxy", "regen_opportunity_index", "fe2c_efficiency_score",
        "classified", "dnf_reason", "rag_context_flag", "data_source",
    ]
    total = 0
    for season_num, gen_id in SEASON_INDEX.values():
        stints = _build_stints(season_num, gen_id)
        if stints:
            pd.DataFrame(stints)[cols].to_sql(
                "race_stints", conn, if_exists="append", index=False)
            total += len(stints)
        time.sleep(0.1)
    print(f"  Loaded {total} stint records across {len(SEASON_INDEX)} seasons.")


# Pipeline Layer 1c: Open-Meteo weather enrichment
# Weather is a confounding variable for efficiency metrics a Monaco race in 35°C heat
# stresses battery thermal management in ways a Berlin race in mild spring temperatures
# does not. Enriching each (track, race_date) pair with historical weather lets downstream
# queries normalize for conditions rather than treating all circuits equally.
# One API call per unique (track, race_date) pair the NOT EXISTS subquery prevents
# re-fetching pairs already in dim_weather, making reruns idempotent.

def _fetch_one_day_weather(lat: float, lon: float, date: str) -> dict | None:
    """Fetch a single circuit-date from Open-Meteo. Returns None on any failure."""
    try:
        resp = requests.get(WEATHER_URL, params={
            "latitude": lat, "longitude": lon,
            "start_date": date, "end_date": date,
            "daily": ",".join(WEATHER_VARS), "timezone": "auto",
        }, timeout=15)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        if not daily or not daily.get("time"):
            return None

        def first(key: str):
            v = daily.get(key, [None])
            return v[0] if v else None

        temp_max = first("temperature_2m_max")
        precip   = first("precipitation_sum")
        wind     = first("wind_speed_10m_max")

        if   precip   is not None and precip   > 2.0:  condition = "Rain"
        elif precip   is not None and precip   > 0.1:  condition = "Light Rain"
        elif temp_max is not None and temp_max > 35:   condition = "Hot/Clear"
        elif wind     is not None and wind     > 40:   condition = "Windy"
        else:                                           condition = "Clear"

        return {
            "temp_avg_c": first("temperature_2m_mean"), "temp_max_c": temp_max,
            "humidity_pct": first("relative_humidity_2m_max"),
            "wind_speed_kmh": wind, "precipitation_mm": precip,
            "condition_label": condition,
        }
    except requests.RequestException as e:
        print(f"    WARNING: Weather fetch failed ({lat},{lon} {date}): {e}")
        return None


def run_weather_ingestion(conn: sqlite3.Connection) -> None:
    """Fetch Open-Meteo weather for each unprocessed (track, race_date) pair."""
    print("  Weather enrichment (Open-Meteo)...")
    # One query returns only pairs not yet in dim_weather, avoids per-row SELECT.
    rows = conn.execute("""
        SELECT DISTINCT t.track_id, t.latitude, t.longitude, s.race_date
        FROM race_stints s
        JOIN dim_track t ON s.track_id = t.track_id
        WHERE t.latitude IS NOT NULL
          AND s.race_date IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM dim_weather w
              WHERE w.track_id = t.track_id AND w.race_date = s.race_date
          )
        ORDER BY s.race_date
    """).fetchall()

    loaded = 0
    for row in rows:
        track_id, lat, lon, date = row["track_id"], row["latitude"], row["longitude"], row["race_date"]
        weather = _fetch_one_day_weather(lat, lon, date)
        if not weather:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO dim_weather "
            "(track_id,race_date,temp_avg_c,temp_max_c,humidity_pct,"
            "wind_speed_kmh,precipitation_mm,condition_label) VALUES (?,?,?,?,?,?,?,?)",
            (track_id, date, weather["temp_avg_c"], weather["temp_max_c"],
             weather["humidity_pct"], weather["wind_speed_kmh"],
             weather["precipitation_mm"], weather["condition_label"]))
        conn.execute(
            "UPDATE race_stints SET weather_id=("
            "SELECT weather_id FROM dim_weather WHERE track_id=? AND race_date=?"
            ") WHERE track_id=? AND race_date=?",
            (track_id, date, track_id, date))
        conn.commit()
        loaded += 1
        time.sleep(0.5)
    print(f"  Loaded {loaded} weather records.")


# Pipeline Layer 1d: California EV registration ingestion
# California is the primary market context for the FE2C consumer simulation (Technical
# Report §6.1) because it has the highest EV adoption rate in the US and two distinct
# driving environments that stress-test different battery strategies: the Bay Area / LA
# Metro corridor (dense regen cycles, short trips) vs. the Inland Empire / Central Valley
# corridor (thermal stress, long haul, sparse charging).
#
# Source hierarchy:
#   1.Live CA DMV open data API — used when available (currently returns 2018 snapshot)
#   2._lucid_supplement() — injected when live data predates Lucid deliveries (Q4 2021)
#   3._build_synthetic_ca_data() — full synthetic fallback if the API is unreachable
#
# INSERT OR IGNORE + executemany replaces pandas to_sql() to handle the UNIQUE constraint
# on (zip_code, brand_raw, model_year, data_year) the DMV source can produce multiple
# rows per key when BEV and PHEV share the same brand/zip/year combination.

def _build_synthetic_ca_data() -> pd.DataFrame:
    """Reproducible synthetic CA registrations used when the live DMV download fails."""
    random.seed(2024)
    brands_bev  = ["TESLA","CHEVROLET","NISSAN","FORD","RIVIAN","HYUNDAI",
                   "KIA","BMW","PORSCHE","VOLKSWAGEN","LUCID"]
    brands_phev = ["TOYOTA","FORD","CHRYSLER","BMW","JEEP","MITSUBISHI","HYUNDAI","VOLVO"]
    records = []
    for zip_code, (city, county) in TARGET_REGIONS.items():
        for brand in brands_bev:
            for year in [2021, 2022, 2023]:
                base = (12 if county == "Los Angeles" else 4) if brand == "LUCID" \
                       else (180 if county == "Los Angeles" else 60)
                records.append({
                    "zip_code": zip_code, "county": county, "city": city,
                    "brand_raw": brand, "model_year": year, "powertrain_type": "BEV",
                    "registration_count": max(1, int(random.gauss(base, base * 0.4))),
                    "data_year": year,
                })
        for brand in brands_phev:
            for year in [2021, 2022, 2023]:
                base = 90 if county == "Los Angeles" else 30
                records.append({
                    "zip_code": zip_code, "county": county, "city": city,
                    "brand_raw": brand, "model_year": year, "powertrain_type": "PHEV",
                    "registration_count": max(1, int(random.gauss(base, base * 0.35))),
                    "data_year": year,
                })
    df = pd.DataFrame(records)
    df.to_csv(RAW_DIR / "ca_ev_registrations_synthetic.csv", index=False)
    return df


def _lucid_supplement() -> pd.DataFrame:
    """Synthetic Lucid registrations for 2022-2023, injected when the live DMV dataset
    predates Lucid deliveries (the CA DMV endpoint currently returns 2018 data only).

    Counts are calibrated to known CA market patterns from Lucid's public delivery reports:
    Bay Area affluent ZIPs account for ~60% of CA Lucid registrations, LA Metro ~30%,
    inland/rural ~10%. Atherton (94027) has the highest per-ZIP density in the state.
    This is explicitly an engineered proxy per Technical Report §8 — logged transparently.
    """
    random.seed(2025)
    # Base registrations per county per ZIP, per year derived from published market data.
    county_base: dict[str, int] = {
        "San Mateo":      48,   # Atherton + Menlo Park + San Carlos: Lucid's strongest CA market
        "Santa Clara":    36,   # Palo Alto + Cupertino: tech demographic, high EV adoption
        "San Francisco":  20,   # High density but smaller individual ZIPs
        "Los Angeles":    16,   # Beverly Hills / Brentwood corridor
        "Riverside":       3,
        "Kern":            2,
        "Fresno":          1,
        "San Bernardino":  1,
        "Stanislaus":      0,
    }
    records = []
    for zip_code, (city, county) in TARGET_REGIONS.items():
        base = county_base.get(county, 2)
        if base == 0:
            continue
        # Atherton is the primary Lucid hotspot double the county base.
        if zip_code == "94027":
            base = base * 2
        for year in [2022, 2023]:
            count = max(1, int(random.gauss(base, base * 0.3)))
            records.append({
                "zip_code": zip_code, "county": county, "city": city,
                "brand_raw": "LUCID", "model_year": year,
                "powertrain_type": "BEV", "registration_count": count,
                "data_year": year,
                "manufacturer_id": LUCID_MFR_ID, "lucid_motors_flag": 1,
            })
    return pd.DataFrame(records)


def _normalize_dmv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Map live DMV column names to the project schema regardless of version."""
    print(f"  DMV columns detected: {df.columns.tolist()}")
    col_lower = {c.lower(): c for c in df.columns}

    def pick(*candidates: str) -> str | None:
        for c in candidates:
            if c.lower() in col_lower:
                return col_lower[c.lower()]
        return None

    zip_col   = pick("ZIP Code", "Zip Code", "zip_code", "ZIP")
    make_col  = pick("Make", "make", "Vehicle Make", "brand_raw")
    fuel_col  = pick("Fuel Type", "fuel_type", "Fuel", "fuel_type_raw")
    year_col  = pick("Model Year", "model_year", "Year")
    count_col = pick("Vehicles", "Count", "vehicle_count", "registration_count", "Number of Vehicles")
    date_col  = pick("Date", "date", "Calendar Year", "Data Year", "data_year")

    missing = [n for n, c in [("zip", zip_col), ("make", make_col), ("count", count_col)] if c is None]
    if missing:
        raise ValueError(f"Cannot locate DMV columns for: {missing}. Available: {df.columns.tolist()}")

    df = df.rename(columns={zip_col: "zip_code", make_col: "brand_raw", count_col: "registration_count"})
    if fuel_col:
        df = df.rename(columns={fuel_col: "fuel_type_raw"})
        df["powertrain_type"] = df["fuel_type_raw"].str.upper().str.strip().map(POWERTRAIN_MAP).fillna("ICE")
    else:
        df["powertrain_type"] = "ICE"

    if year_col:
        df = df.rename(columns={year_col: "model_year"})
    if date_col:
        df = df.rename(columns={date_col: "data_year"})
        if df["data_year"].dtype == object:
            df["data_year"] = pd.to_datetime(df["data_year"], errors="coerce").dt.year
        df["data_year"] = pd.to_numeric(df["data_year"], errors="coerce")
    else:
        df["data_year"] = pd.to_numeric(
            df["model_year"] if "model_year" in df.columns else 2023,
            errors="coerce").fillna(2023)

    df["brand_raw"]          = df["brand_raw"].astype(str).str.upper().str.strip()
    df["zip_code"]           = df["zip_code"].astype(str).str.strip().str.zfill(5)
    df["model_year"]         = pd.to_numeric(
        df["model_year"] if "model_year" in df.columns else 0,
        errors="coerce").fillna(0).astype(int)
    df["registration_count"] = pd.to_numeric(df["registration_count"], errors="coerce").fillna(0).astype(int)
    return df


def run_california_ingestion(conn: sqlite3.Connection) -> None:
    """Load CA EV registrations — live DMV preferred, reproducible synthetic fallback."""
    print("  California EV registration data...")
    df = None
    try:
        resp = requests.get(CA_DMV_URL, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
        df.to_csv(RAW_DIR / "ca_dmv_vehicle_fuel_type.csv", index=False)
        print(f"  Live DMV data: {len(df):,} rows")
        df = _normalize_dmv_dataframe(df)
        df = df[df["zip_code"].isin(TARGET_REGIONS)]
        df = df[df["powertrain_type"].isin(["BEV", "PHEV", "FCEV"])]
        df = df[df["registration_count"] > 0]
        print(f"  After filter: {len(df):,} rows")
        if df.empty:
            raise ValueError("Zero rows after filter.")
        df["city"]            = df["zip_code"].map(lambda z: TARGET_REGIONS.get(z, ("",""))[0])
        df["county"]          = df["zip_code"].map(lambda z: TARGET_REGIONS.get(z, ("",""))[1])
        df["manufacturer_id"] = df["brand_raw"].map(BRAND_TO_MFR)
    except Exception as e:
        print(f"  WARNING: Live DMV failed ({e}) — using synthetic data.")
        df = _build_synthetic_ca_data()
        df["manufacturer_id"] = df["brand_raw"].map(BRAND_TO_MFR)

    if df is None or df.empty:
        print("  WARNING: No CA data to load.")
        return

    for col in ["zip_code", "county", "city", "manufacturer_id", "brand_raw",
                "model_year", "powertrain_type", "registration_count", "data_year"]:
        if col not in df.columns:
            df[col] = None

    df["data_year"]        = df["data_year"].fillna(0).astype(int)
    df["lucid_motors_flag"] = (df["brand_raw"] == "LUCID").astype(int)

    # The CA DMV endpoint currently returns a 2018 snapshot Lucid's first delivery was
    # Q4 2021, so it structurally cannot appear in that dataset. When the live data shows
    # zero Lucid registrations, inject a calibrated synthetic supplement (Technical Report §8)
    # so the market penetration simulation has meaningful signal for the FE2C thesis.
    if df["lucid_motors_flag"].sum() == 0:
        lucid_df = _lucid_supplement()
        lucid_df["data_year"]        = lucid_df["data_year"].fillna(0).astype(int)
        lucid_df["lucid_motors_flag"] = 1
        df = pd.concat([df, lucid_df], ignore_index=True)
        print(f"  Lucid supplement injected: {len(lucid_df)} rows "
              f"(live DMV dataset predates 2021 commercial deliveries).")

    # Aggregate before insert the DMV source can have multiple powertrain rows
    # per (zip, brand, model_year, data_year) that map to the same UNIQUE key.
    key_cols  = ["zip_code", "brand_raw", "model_year", "data_year", "powertrain_type"]
    meta_cols = ["county", "city", "manufacturer_id", "lucid_motors_flag"]
    df = (
        df.groupby(key_cols, as_index=False)
          .agg(registration_count=("registration_count", "sum"),
               **{c: (c, "first") for c in meta_cols})
    )

    final_cols   = ["zip_code", "county", "city", "manufacturer_id", "brand_raw",
                    "model_year", "powertrain_type", "registration_count",
                    "data_year", "lucid_motors_flag"]
    cols_str     = ", ".join(final_cols)
    placeholders = ", ".join(["?"] * len(final_cols))
    rows = df[final_cols].where(df[final_cols].notna(), None).values.tolist()
    conn.executemany(
        f"INSERT OR IGNORE INTO ca_ev_registrations ({cols_str}) VALUES ({placeholders})", rows)
    conn.commit()
    print(f"  Loaded {len(df):,} CA registration records.")


# Pipeline Layer 1e: Derived efficiency metrics
# Three metrics are computed here and written back to race_stints. They cannot be
# computed at insert time because they depend on aggregates across the full dataset.
#
#   regen_opportunity_index: how much regen opportunity the circuit + conditions created,
#     normalized by pack size so Gen 2 and Gen 3 are on a comparable scale. Not live
#     telemetry — a proxy engineered from elevation and lap variance (Technical Report §6.2).
#
#   fe2c_efficiency_score: 70% energy efficiency + 30% normalized velocity, min-max
#     scaled within each generation cohort. The 70/30 split ensures energy discipline
#     drives the ranking rather than raw pace (Technical Report §5.2).
#
#   delta_e: energy cost proxy minus the race-average cost. Negative = consumed less
#     energy than the field that race = efficient battery management (Technical Report §5.1).
#     This is the primary per-stint transfer signal for the FE2C thesis.

def compute_efficiency_metrics(conn: sqlite3.Connection) -> None:
    """Compute regen_opportunity_index, fe2c_efficiency_score, and delta_e for all stints.

    regen_opportunity_index: (1/lap_var) * log1p(elevation_delta) / battery_kwh
      — a proxy for how much regenerative braking opportunity the circuit + driver combo
        created, normalised by pack size so Gen2 and Gen3 are on a comparable scale.

    fe2c_efficiency_score: min-max within the generation cohort, weighted 70% energy efficiency
      / 30% normalized velocity — PDF §5.2 defines this as efficiency-first, not pace-first.
      Raw velocity is capped to a normalized [0,1] contribution so a slow team with excellent
      energy discipline doesn't rank below a fast team with wasteful management.

    delta_e (Technical Report §5.1): energy_cost_proxy minus the race-average cost.
      Negative delta_e = team consumed less energy than the field that race = efficient
      battery management. This is the key per-stint signal for the FE2C thesis.
    """
    print("  Computing efficiency metrics...")
    df = pd.read_sql_query("""
        SELECT s.stint_id, s.lap_time_variance, s.race_distance_km,
               s.avg_lap_velocity_kmh, s.eer_proxy,
               s.season, s.race_round,
               t.elevation_delta_m, g.battery_kwh, g.gen_name
        FROM race_stints s
        JOIN dim_track t       ON s.track_id = t.track_id
        JOIN dim_battery_gen g ON s.gen_id   = g.gen_id
        WHERE s.lap_time_variance > 0
          AND t.elevation_delta_m IS NOT NULL
          AND s.eer_proxy IS NOT NULL
          AND s.classified = 1
    """, conn)

    if df.empty:
        # If the DB has no eligible stints at all, fall back to the engineered proxy
        # values so downstream agent tools still return directional estimates.
        print(
            f"  WARNING: No eligible stints for metric computation.\n"
            f"  Engineered proxy assumptions (Technical Report §8):\n"
            f"    Gen3 recovery ratio = {ENGINEERED_PROXIES['gen3_recovery_ratio']:.0%}\n"
            f"    Field energy draw   = {ENGINEERED_PROXIES['field_energy_kwh_per_km']} kWh/km\n"
            f"  Run --reset then the full pipeline to replace proxies with real data."
        )
        return

    df["regen_opportunity_index"] = (
        (1.0 / df["lap_time_variance"])
        * df["elevation_delta_m"].apply(math.log1p)
        / df["battery_kwh"]
    ).round(6)

    # Velocity-normalized FE2C score per PDF §5.2: efficiency drives 70% of the score,
    # pace only 30%, capped via within-cohort min-max so a fast-but-wasteful team can't
    # outrank a slower team with superior energy discipline.
    vel_norm = df.groupby("gen_name")["avg_lap_velocity_kmh"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9)
    )
    df["raw_fe2c"] = (
        (df["race_distance_km"] / df["eer_proxy"].clip(lower=1e-6)) * 0.70
        + vel_norm * 0.30
    )

    def minmax(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        return pd.Series(50.0, index=s.index) if hi - lo < 1e-9 \
               else (s - lo) / (hi - lo) * 100

    df["fe2c_efficiency_score"] = df.groupby("gen_name")["raw_fe2c"].transform(minmax).round(2)

    # delta_e: invert EER to get an energy-cost proxy (lower EER = more energy per unit work),
    # then subtract the per-race mean so the sign conveys direction against the field.
    df["energy_cost_proxy"] = 1.0 / df["eer_proxy"].clip(lower=1e-6)
    race_avg_cost = df.groupby(["season", "race_round"])["energy_cost_proxy"].transform("mean")
    df["delta_e"] = (df["energy_cost_proxy"] - race_avg_cost).round(6)

    conn.executemany(
        "UPDATE race_stints "
        "SET regen_opportunity_index=?, fe2c_efficiency_score=?, delta_e=? "
        "WHERE stint_id=?",
        list(zip(
            df["regen_opportunity_index"],
            df["fe2c_efficiency_score"],
            df["delta_e"],
            df["stint_id"],
        )))
    conn.commit()
    total_stints = conn.execute("SELECT COUNT(*) FROM race_stints").fetchone()[0]
    print(f"  Metrics computed for {len(df):,} classified stints (of {total_stints:,} total).")


# Pipeline Layer 1f: Data quality assertions.
# Ten targeted checks run after metrics are computed. These are not unit tests they
# are pipeline integrity checks that confirm the data meets the assumptions the Week 3
# analytics depend on. Each check returns a violation count rather than a boolean so
# the severity is visible at a glance. A production version of this would be a dbt test
# layer (Technical Report §4.3) but SQL-in-Python is sufficient for a single-file project.

def run_quality_checks(conn: sqlite3.Connection) -> dict:
    """Ten targeted QA assertions. Returns summary dict; prints PASS/FAIL per check."""
    checks = []

    def check(name: str, query: str) -> None:
        n      = conn.execute(query).fetchone()[0]
        status = "PASS" if n == 0 else "FAIL"
        checks.append({"name": name, "count": n, "status": status})
        print(f"  [{status}] {name}: {n} violations")

    check("eer_proxy populated",
          "SELECT COUNT(*) FROM race_stints WHERE eer_proxy IS NULL AND classified=1")
    check("eer_proxy formula accuracy",
          "SELECT COUNT(*) FROM race_stints WHERE "
          "ABS(eer_proxy-(race_distance_km/(avg_lap_sec*lap_time_variance)))>0.001 AND classified=1")
    check("fe2c_score populated",
          "SELECT COUNT(*) FROM race_stints WHERE fe2c_efficiency_score IS NULL AND classified=1")
    check("finish_position <= 16",
          "SELECT COUNT(*) FROM race_stints WHERE finish_position > 16")
    check("grid_position <= 16",
          "SELECT COUNT(*) FROM race_stints WHERE grid_position > 16")
    check("positions_gained zero-sum per race",
          "SELECT COUNT(*) FROM (SELECT season, race_round, SUM(positions_gained) t "
          "FROM race_stints GROUP BY season, race_round HAVING ABS(t)>1)")
    check("no orphaned track_id",
          "SELECT COUNT(*) FROM race_stints s LEFT JOIN dim_track t ON s.track_id=t.track_id "
          "WHERE s.track_id IS NOT NULL AND t.track_id IS NULL")
    check("classified/dnf consistency",
          "SELECT COUNT(*) FROM race_stints WHERE classified=1 AND dnf_reason IS NOT NULL")
    check("fastest_lap < avg_lap",
          "SELECT COUNT(*) FROM race_stints WHERE fastest_lap_sec >= avg_lap_sec")
    check("regen_index populated",
          "SELECT COUNT(*) FROM race_stints WHERE regen_opportunity_index IS NULL AND classified=1")

    failed = sum(1 for c in checks if c["status"] == "FAIL")
    print(f"  {len(checks) - failed}/{len(checks)} checks passed.")
    return {"passed": len(checks) - failed, "failed": failed}


# Pipeline Layer 1g: Week 3 consumer market simulation
# Three analytical layers compose the Week 3 output:
#
#   1. Racing efficiency ranking: manufacturers ranked by FE2C score with 95% CI,
#      frontier classification (Efficiency Frontier / Fast Low Regen / etc.), and
#      Lucid chain tagging. This is the primary evidence table for the thesis.
#
#   2. CA market penetration: EV registration share by brand and county, contrasting
#      the Bay Area (dense regen) with the inland corridor (thermal stress). Lucid's
#      share in tech-affluent ZIPs like Atherton is the consumer market signal.
#
#   3. Range transfer simulation: bootstrapped estimate of how Lucid/Mahindra's Gen 3
#      regen index advantage translates to real-world range uplift on the Air Sapphire,
#      Grand Touring, and Gravity. Uses 1,000 bootstrap resamples for a 95% CI so the
#      uncertainty is quantified, not hidden.

def week3_preflight(conn: sqlite3.Connection) -> None:
    checks = {
        "race_stints rows":      "SELECT COUNT(*) FROM race_stints",
        "fe2c_score populated":  "SELECT COUNT(*) FROM race_stints WHERE fe2c_efficiency_score IS NOT NULL",
        "regen_index populated": "SELECT COUNT(*) FROM race_stints WHERE regen_opportunity_index IS NOT NULL",
        "ca_ev_registrations":   "SELECT COUNT(*) FROM ca_ev_registrations",
    }
    for label, q in checks.items():
        n = conn.execute(q).fetchone()[0]
        if n == 0:
            sys.exit(f"Week 3 FAIL: {label} = 0. Run full pipeline first.")
        print(f"  {label}: {n:,}")


def compute_racing_efficiency(conn: sqlite3.Connection) -> pd.DataFrame:
    """Rank manufacturers by FE2C score and regen index with 95% CI and frontier class."""
    df = pd.read_sql_query("""
        SELECT m.manufacturer_name, m.ca_market_presence, m.consumer_ev_lineup,
               g.gen_name, s.season, s.fe2c_efficiency_score,
               s.regen_opportunity_index, s.eer_proxy, s.avg_lap_velocity_kmh,
               s.positions_gained, s.classified,
               CASE WHEN s.manufacturer_id IN (?,?) THEN 1 ELSE 0 END AS lucid_chain
        FROM race_stints s
        JOIN dim_manufacturer m ON s.manufacturer_id = m.manufacturer_id
        JOIN dim_battery_gen  g ON s.gen_id = g.gen_id
        WHERE s.classified = 1
          AND s.fe2c_efficiency_score IS NOT NULL
          AND s.regen_opportunity_index IS NOT NULL
    """, conn, params=(LUCID_MFR_ID, MAHINDRA_MFR_ID))

    if df.empty:
        sys.exit("Week 3 FAIL: No classified stints with computed scores.")

    agg = df.groupby("manufacturer_name").agg(
        avg_fe2c_score       =("fe2c_efficiency_score",   "mean"),
        std_fe2c_score       =("fe2c_efficiency_score",   "std"),
        avg_regen_index      =("regen_opportunity_index", "mean"),
        avg_eer_proxy        =("eer_proxy",               "mean"),
        avg_velocity         =("avg_lap_velocity_kmh",    "mean"),
        avg_positions_gained =("positions_gained",        "mean"),
        total_stints         =("classified",              "count"),
        seasons_active       =("season",                  "nunique"),
        ca_market_presence   =("ca_market_presence",      "first"),
        consumer_ev_lineup   =("consumer_ev_lineup",      "first"),
        lucid_chain          =("lucid_chain",             "first"),
    ).reset_index()

    agg["fe2c_score_ci95"] = (1.96 * agg["std_fe2c_score"] / np.sqrt(agg["total_stints"])).round(2)
    agg["efficiency_rank"] = agg["avg_fe2c_score"].rank(ascending=False).astype(int)

    vel_avg, regen_avg = agg["avg_velocity"].mean(), agg["avg_regen_index"].mean()
    agg["frontier_class"] = np.select(
        [
            (agg["avg_velocity"] >= vel_avg) & (agg["avg_regen_index"] >= regen_avg),
            (agg["avg_velocity"] >= vel_avg) & (agg["avg_regen_index"] < regen_avg),
            (agg["avg_velocity"] < vel_avg)  & (agg["avg_regen_index"] >= regen_avg),
        ],
        ["Efficiency Frontier", "Fast / Low Regen", "Consistent / Slow"],
        default="Below Average",
    )
    return agg.sort_values("efficiency_rank").round(4)


def compute_ca_market_penetration(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query("""
        SELECT r.county, r.brand_raw, r.lucid_motors_flag,
               r.registration_count, m.manufacturer_name, m.ca_market_presence
        FROM ca_ev_registrations r
        LEFT JOIN dim_manufacturer m ON r.manufacturer_id = m.manufacturer_id
        WHERE r.powertrain_type IN ('BEV','PHEV','FCEV') AND r.registration_count > 0
    """, conn)

    if df.empty:
        sys.exit("Week 3 FAIL: ca_ev_registrations empty after filter.")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    county_totals = df.groupby("county")["registration_count"].sum().rename("county_total")
    df = df.merge(county_totals, on="county", how="left")
    df["brand_county_regs"] = df.groupby(["county", "brand_raw"])["registration_count"].transform("sum")
    df["market_share_pct"]  = (df["brand_county_regs"] / df["county_total"].clip(lower=1) * 100).round(3)

    result = (
        df.groupby(["county", "brand_raw"]).agg(
            total_regs        =("registration_count", "sum"),
            county_total      =("county_total",        "first"),
            market_share_pct  =("market_share_pct",    "first"),
            lucid_flag        =("lucid_motors_flag",   "first"),
            manufacturer_name =("manufacturer_name",   "first"),
        ).reset_index()
         .sort_values(["county", "market_share_pct"], ascending=[True, False])
    )
    result["is_lucid"] = result["lucid_flag"] == 1

    share_sums   = result.groupby("county")["market_share_pct"].sum()
    bad_counties = share_sums[share_sums < 95].index.tolist()
    if bad_counties:
        print(f"  WARNING: market shares < 95% in {bad_counties} — check registration data.")
    return result


def compute_range_simulation(conn: sqlite3.Connection) -> dict:
    """
    Estimate Lucid Air real-world range using FE regen performance as a transfer proxy.
    Transfer factor = (lucid_regen - field_regen) / field_regen, clipped to [0, 2].
    Bootstrap CI uses 1000 vectorized numpy resamples.
    """
    df = pd.read_sql_query("""
        SELECT s.regen_opportunity_index, s.fe2c_efficiency_score,
               m.manufacturer_id
        FROM race_stints s
        JOIN dim_manufacturer m ON s.manufacturer_id = m.manufacturer_id
        JOIN dim_battery_gen  g ON s.gen_id = g.gen_id
        WHERE s.classified = 1
          AND s.regen_opportunity_index IS NOT NULL
          AND s.fe2c_efficiency_score IS NOT NULL
          AND g.gen_name = 'Gen3'
    """, conn)

    if df.empty:
        return {"error": "No Gen3 stints with regen data."}

    lucid_mask  = df["manufacturer_id"].isin([LUCID_MFR_ID, MAHINDRA_MFR_ID])
    lucid_regen = df.loc[lucid_mask, "regen_opportunity_index"].values
    field_regen = df.loc[~lucid_mask, "regen_opportunity_index"].values

    if len(lucid_regen) == 0 or len(field_regen) == 0:
        return {"error": "Insufficient data for comparison."}

    lucid_mean      = lucid_regen.mean()
    field_mean      = field_regen.mean()
    t_stat, p_val   = stats.ttest_ind(lucid_regen, field_regen, equal_var=False)
    transfer_factor = float(np.clip((lucid_mean - field_mean) / max(field_mean, 1e-9), 0, 2.0))
    sim_range       = LUCID_AIR_EPA_RANGE * (1 + transfer_factor * REGEN_WEIGHT)

    np.random.seed(42)
    boot_idx     = np.random.choice(len(field_regen), size=(1000, len(field_regen)), replace=True)
    boot_means   = field_regen[boot_idx].mean(axis=1)
    boot_factors = np.clip((lucid_mean - boot_means) / np.maximum(boot_means, 1e-9), 0, 2.0)
    boot_ranges  = LUCID_AIR_EPA_RANGE * (1 + boot_factors * REGEN_WEIGHT)
    ci_lo, ci_hi = float(np.percentile(boot_ranges, 2.5)), float(np.percentile(boot_ranges, 97.5))

    lucid_fe2c = df.loc[lucid_mask,  "fe2c_efficiency_score"].mean()
    field_fe2c = df.loc[~lucid_mask, "fe2c_efficiency_score"].mean()

    return {
        "lucid_avg_regen":    round(float(lucid_mean), 6),
        "field_avg_regen":    round(float(field_mean), 6),
        "transfer_factor":    round(transfer_factor, 4),
        "epa_baseline_mi":    LUCID_AIR_EPA_RANGE,
        "simulated_range_mi": round(float(sim_range), 1),
        "ci_lo": round(ci_lo, 1), "ci_hi": round(ci_hi, 1),
        "t_stat": round(float(t_stat), 3), "p_value": round(float(p_val), 4),
        "significant": bool(p_val < 0.05),
        "lucid_fe2c_score": round(float(lucid_fe2c), 2),
        "field_fe2c_score": round(float(field_fe2c), 2),
        "n_lucid": int(lucid_mask.sum()), "n_field": int((~lucid_mask).sum()),
    }


def print_week3_report(eff: pd.DataFrame, pen: pd.DataFrame, sim: dict) -> None:
    print("\nFE2C WEEK 3 — CA CONSUMER MARKET SIMULATION REPORT")

    if "error" not in sim:
        # PDF §8 requires the significance flag to appear at point-of-claim, not buried
        # in the adversarial note a reader scanning the BLUF must see the caveat
        # before they accept the simulated range as a reliable number.
        sig_flag  = "NOT SIGNIFICANT — treat as directional only" if not sim["significant"] \
                    else "STATISTICALLY SIGNIFICANT (p<0.05)"
        sig_label = "not significant (small sample)" if not sim["significant"] \
                    else "statistically significant"
        print(
            f"\nBLUF: Lucid/Mahindra Gen3 regen index ({sim['lucid_avg_regen']:.4f}) vs "
            f"field ({sim['field_avg_regen']:.4f}) yields a {sim['transfer_factor']*100:.1f}% "
            f"regen advantage. Applied to the Lucid Air EPA baseline of {sim['epa_baseline_mi']} mi, "
            f"simulated range = {sim['simulated_range_mi']} mi "
            f"(95% CI: {sim['ci_lo']}–{sim['ci_hi']} mi). "
            f"t={sim['t_stat']}, p={sim['p_value']} — {sig_label}.\n"
            f"  [{sig_flag}]"
        )
        print(
            f"\n  Methodology note (PDF §6.2 + §8): regen_opportunity_index is a proxy "
            f"derived from lap variance and elevation — not live telemetry. The 15% regen "
            f"weight and transfer factor are assumptions, not calibrated parameters. "
            f"n_lucid={sim['n_lucid']}, n_field={sim['n_field']}."
        )
    else:
        print(f"\nBLUF: Range simulation failed — {sim['error']}")

    print("\nRACING EFFICIENCY RANKING")
    print(f"  {'Rank':<5} {'Manufacturer':<22} {'FE2C Score':>10} {'±CI95':>7} {'Regen Index':>12} {'Frontier'}")
    for _, row in eff.iterrows():
        tag = "  [LUCID CHAIN]" if row["lucid_chain"] else ""
        print(f"  {int(row['efficiency_rank']):<5} {row['manufacturer_name']:<22} "
              f"{row['avg_fe2c_score']:>10.2f} {row['fe2c_score_ci95']:>7.2f} "
              f"{row['avg_regen_index']:>12.6f} {row['frontier_class']}{tag}")

    print("\nCA MARKET PENETRATION (top 5 per county)")
    for county, grp in pen.groupby("county"):
        lucid_rows  = grp[grp["is_lucid"]]
        lucid_share = lucid_rows["market_share_pct"].values[0] if not lucid_rows.empty else 0.0
        lucid_rank  = (
            int(grp["market_share_pct"].rank(ascending=False).loc[lucid_rows.index].values[0])
            if not lucid_rows.empty else "N/A"
        )
        print(f"\n  {county}  (Lucid: {lucid_share:.3f}% — #{lucid_rank})")
        for _, br in grp.nlargest(5, "market_share_pct").iterrows():
            tag = "  [LUCID]" if br["is_lucid"] else ""
            print(f"    {br['brand_raw']:<20} {br['market_share_pct']:>7.3f}%{tag}")

    if "error" not in sim:
        print(f"\nRANGE TRANSFER SIMULATION")
        print(f"  Lucid/Mahindra Gen3 stints: {sim['n_lucid']}  |  Field Gen3 stints: {sim['n_field']}")
        print(f"  Lucid avg regen:  {sim['lucid_avg_regen']:.6f}  |  Field avg regen: {sim['field_avg_regen']:.6f}")
        print(f"  Transfer factor:  {sim['transfer_factor']:.4f} ({sim['transfer_factor']*100:.1f}% advantage)")
        print(f"  Regen weight:     {REGEN_WEIGHT} (15% — conservative midpoint of real-world regen contribution)")
        print(f"  EPA baseline:     {sim['epa_baseline_mi']} mi")
        print(f"  Simulated range:  {sim['simulated_range_mi']} mi  (95% CI: {sim['ci_lo']}–{sim['ci_hi']} mi)")
        print(f"  FE2C score — Lucid chain: {sim['lucid_fe2c_score']:.2f}  |  Field: {sim['field_fe2c_score']:.2f}")

    # Consumer benchmark crosswalk: express the simulated range uplift across every
    # Lucid model so Week 3 answers "how does Gen3 racing translate to your driveway?"
    # This is the core FE2C thesis made concrete, Technical Report §6.3.
    transfer_pct = sim.get("transfer_factor", 0.0) * REGEN_WEIGHT
    print(f"\nLUCID CONSUMER BENCHMARK CROSSWALK (Gen3 regen uplift applied)")
    print(f"  {'Model':<22} {'EPA Range':>10} {'Simulated':>10} {'Battery':>9}  Segment")
    for model_name, spec in LUCID_BENCHMARKS.items():
        sim_r = spec["epa_range_mi"] * (1 + transfer_pct)
        print(
            f"  {model_name:<22} {spec['epa_range_mi']:>9} mi"
            f" {sim_r:>9.1f} mi {spec['battery_kwh']:>7.1f} kWh  {spec['segment']}"
        )
    print(f"  Note: {spec['note'] if LUCID_BENCHMARKS else ''}")

    print(
        "\nADVERSARIAL NOTE: regen_opportunity_index is a proxy derived from lap variance "
        "and elevation — not live telemetry. Formula E does not publish energy recovery data "
        "publicly. The transfer factor may reflect circuit characteristics rather than genuine "
        "powertrain superiority. The 15% regen weight is an assumption, not a calibrated "
        "parameter. CA market share figures use synthetic registrations if the live DMV download failed."
    )


def save_week3_outputs(eff: pd.DataFrame, pen: pd.DataFrame, sim: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eff.to_csv(OUT_DIR / "racing_efficiency.csv", index=False)
    pen.to_csv(OUT_DIR / "ca_market_penetration.csv", index=False)
    pd.DataFrame([sim]).to_csv(OUT_DIR / "range_simulation.csv", index=False)
    print(f"\n  CSVs saved to {OUT_DIR}/")


def run_week3(conn: sqlite3.Connection, save_csv: bool = True) -> None:
    """Execute all three Week 3 analytical layers: efficiency, penetration, simulation."""
    print("\n[Week 3] Preflight checks...")
    week3_preflight(conn)
    print("[Week 3] Racing efficiency ranking...")
    eff = compute_racing_efficiency(conn)
    print("[Week 3] CA market penetration...")
    pen = compute_ca_market_penetration(conn)
    print("[Week 3] Range transfer simulation...")
    sim = compute_range_simulation(conn)
    print_week3_report(eff, pen, sim)
    if save_csv:
        save_week3_outputs(eff, pen, sim)


# Layer 2: RAG: ChromaDB ingestion, retrieval, and outlier enrichment
# The vector store exists because the structured SQLite database cannot answer qualitative
# questions: "What did Mahindra's engineering team change between seasons?" or "What does
# the Gen 3 front axle regen spec mean for consumer charging?" Those answers live in text 
# the Technical Report, Wikipedia, and team press releases. ChromaDB bridges the two layers.
#
# Retrieval uses cosine similarity on sentence-transformer embeddings (all-MiniLM-L6-v2).
# Chunk size 600 chars with 100-char overlap preserves sentence context across boundaries.
# The outlier enrichment step (--enrich) identifies stints that are statistical outliers
# (|z-score| > 2.0) and asks Claude to explain them using the retrieved context this
# populates the rag_context table, which the agent can then query as structured data.

def chunk_text(text: str, source_name: str) -> list[dict]:
    chunks: list[dict] = []
    text  = text.strip()
    start = 0
    idx   = 0
    while start < len(text):
        end   = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append({
                "text": chunk, "id": f"{source_name}__chunk_{idx}",
                "metadata": {"source": source_name, "chunk_index": idx, "char_start": start},
            })
            idx += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def fetch_wikipedia_text(url: str) -> str | None:
    try:
        title   = url.rstrip("/").split("/wiki/")[-1]
        headers = {"User-Agent": "FE2C-RAG/1.0"}
        r1 = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
            timeout=15, headers=headers)
        r1.raise_for_status()
        summary = r1.json().get("extract", "")
        r2 = requests.get("https://en.wikipedia.org/w/api.php",
            params={"action": "query", "titles": title.replace("_", " "),
                    "prop": "extracts", "explaintext": True, "format": "json"},
            timeout=15, headers=headers)
        r2.raise_for_status()
        pages = r2.json().get("query", {}).get("pages", {})
        return next(iter(pages.values()), {}).get("extract", summary) or summary
    except Exception as e:
        print(f"    WARNING: Failed to fetch {url}: {e}")
        return None


def _extract_pdf_text(pdf_path: Path) -> str | None:
    """Extract plain text from a PDF using pypdf.

    pypdf is a pure-Python parser — no system dependencies (no poppler, no ghostscript).
    The extracted text is returned as a single string suitable for chunk_text().
    Returns None on any failure so the caller can skip gracefully.
    """
    try:
        import pypdf  # type: ignore[import-untyped]
        reader = pypdf.PdfReader(str(pdf_path))
        pages  = [page.extract_text() or "" for page in reader.pages]
        text   = "\n\n".join(p.strip() for p in pages if p.strip())
        return text if text else None
    except Exception as e:
        print(f"    WARNING: PDF extraction failed for {pdf_path.name}: {e}")
        return None


def run_ingest() -> None:
    """Build or refresh the ChromaDB vector store.

    Source priority (highest signal first):
      1. rag_docs/*.txt  — structured excerpts from the Technical Report; highest retrieval
                           precision because they are written in the project's own vocabulary.
      2. rag_docs/*.pdf  — any PDF dropped into the folder is auto-extracted via pypdf.
      3. Wikipedia       — broad background context; lower domain specificity than §1–2.

    Loading local docs before Wikipedia means ChromaDB's HNSW index sees the most
    domain-relevant text first during construction, biasing nearest-neighbour results
    toward the Technical Report content when queries use project-specific terminology.
    """
    print("[RAG INGEST] Building ChromaDB vector store...")
    collection = _get_chroma_collection()
    existing   = set(collection.get()["ids"]) if collection.count() > 0 else set()

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    all_docs: list[tuple[str, str]] = []

    # Local .txt files first these are the curated Technical Report excerpts that give
    # the agent precise vocabulary for project-specific questions (delta_e, regen index, etc.)
    for f in sorted(DOCS_DIR.glob("*.txt")):
        try:
            all_docs.append((f.stem, f.read_text(encoding="utf-8")))
            print(f"  Loaded .txt: {f.name}")
        except Exception as e:
            print(f"  WARNING: {f.name}: {e}")

    # PDF files in rag_docs/ — auto-extract text so dropping a PDF is enough to ingest it.
    # pypdf is pure Python and already installed alongside anthropic.
    for f in sorted(DOCS_DIR.glob("*.pdf")):
        text = _extract_pdf_text(f)
        if text:
            all_docs.append((f.stem, text))
            print(f"  Loaded .pdf: {f.name} ({len(text):,} chars)")

    # Wikipedia provides broad Formula E and Lucid Motors background. Fetched after local
    # docs so project-specific vocabulary is already seeded in the index.
    for name, url in WIKIPEDIA_SOURCES:
        print(f"  Fetching Wikipedia: {name}")
        text = fetch_wikipedia_text(url)
        if text:
            all_docs.append((name, text))
        time.sleep(0.5)

    total_new = 0
    for source_name, text in all_docs:
        chunks     = chunk_text(text, source_name)
        new_chunks = [c for c in chunks if c["id"] not in existing]
        if new_chunks:
            collection.upsert(
                ids=[c["id"] for c in new_chunks],
                documents=[c["text"] for c in new_chunks],
                metadatas=[c["metadata"] for c in new_chunks])
            total_new += len(new_chunks)
            print(f"  {source_name}: {len(new_chunks)} chunks upserted.")
    print(f"  Done. Total chunks: {collection.count()} (+{total_new} new)")


def retrieve_context(collection: Any, query: str) -> tuple[str, float, str]:
    results = collection.query(
        query_texts=[query], n_results=min(TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"])
    docs, metas, dists = results["documents"][0], results["metadatas"][0], results["distances"][0]
    sims = [round(1 - d / 2, 4) for d in dists]
    return (
        "\n\n---\n\n".join(docs),
        sims[0] if sims else 0.0,
        metas[0].get("source", "unknown") if metas else "unknown",
    )


def run_enrichment() -> None:
    import anthropic as _anthropic  # type: ignore[import-untyped]

    if not DB_PATH.exists():
        sys.exit("ERROR: Database not found. Run full pipeline first.")

    collection = _get_chroma_collection(require_populated=True)
    client     = _anthropic.Anthropic()

    with get_connection() as conn:
        df = pd.read_sql_query("""
            SELECT s.stint_id, m.manufacturer_name, s.season, s.race_round,
                   s.race_name, s.driver_name, s.fe2c_efficiency_score,
                   s.eer_proxy, s.regen_opportunity_index, s.positions_gained,
                   s.avg_lap_velocity_kmh, s.lap_time_variance,
                   g.gen_name, t.circuit_name, t.surface_type, t.elevation_delta_m,
                   w.temp_max_c, w.condition_label
            FROM race_stints s
            JOIN dim_manufacturer m ON s.manufacturer_id = m.manufacturer_id
            JOIN dim_battery_gen  g ON s.gen_id = g.gen_id
            JOIN dim_track        t ON s.track_id = t.track_id
            LEFT JOIN dim_weather w ON s.weather_id = w.weather_id
            WHERE s.classified = 1 AND s.fe2c_efficiency_score IS NOT NULL
              AND s.rag_context_flag IS NULL
        """, conn)

        if df.empty:
            print("  No stints to enrich.")
            return

        df["z_score"] = df.groupby("gen_name")["fe2c_efficiency_score"].transform(
            lambda x: (x - x.mean()) / x.std(ddof=1))
        outliers = df[df["z_score"].abs() > OUTLIER_Z_THRESHOLD].copy()
        print(f"  {len(outliers)} outlier stints to enrich.")

        for _, row in outliers.iterrows():
            stint_id  = int(row["stint_id"])
            direction = "above" if row["z_score"] > 0 else "below"
            #Bias toward battery innovation context Gen3 queries surface thermal management
            #and charge speed themes; Gen2 queries surface energy density and pack engineering.
            #This aligns ChromaDB retrieval with the FE2C thesis rather than generic race recaps.
            gen_framing = (
                "Gen3 active thermal management charge speed front axle regeneration 250kW"
                if row["gen_name"] == "Gen3"
                else "Gen2 energy density single charge full race distance pack engineering"
            )
            query = (
                f"{row['manufacturer_name']} Formula E battery efficiency "
                f"{gen_framing} "
                f"{row['race_name']} season {row['season']} "
                f"{'unusually high regen recovery' if row['z_score'] > 0 else 'low energy efficiency outlier'}"
            )
            context, score, source = retrieve_context(collection, query)
            weather_str = (
                f"{row['condition_label']} ({row['temp_max_c']}°C)"
                if pd.notna(row.get("condition_label")) else "unknown"
            )
            prompt = (
                f"You are an FE2C Technical Lead. Explain in 2-3 sentences why this stint is "
                f"a notable efficiency outlier, framing it in terms of what the battery management "
                f"strategy implies for consumer EV development. Be specific.\n\n"
                f"Team: {row['manufacturer_name']} | Race: {row['race_name']} S{row['season']} "
                f"R{row['race_round']} | Circuit: {row['circuit_name']} "
                f"({row['surface_type']}, {row['elevation_delta_m']}m elevation)\n"
                f"Gen: {row['gen_name']} | Weather: {weather_str}\n"
                f"FE2C Score: {row['fe2c_efficiency_score']:.1f} "
                f"({row['z_score']:.2f}z — {abs(row['z_score']):.1f} std {direction} avg)\n"
                f"EER: {row['eer_proxy']:.4f} | Regen: {row['regen_opportunity_index']:.4f}\n\n"
                f"CONTEXT:\n{context}\n\nNo headers, no bullets, no filler phrases."
            )
            try:
                diagnostic = client.messages.create(
                    model=CLAUDE_MODEL, max_tokens=300,
                    messages=[{"role": "user", "content": prompt}]
                ).content[0].text.strip()
            except Exception as api_err:
                #API failure fallback per Technical Report §8 (Engineered Proxies).
                #Rather than dropping the row, write a structured proxy annotation so the
                #rag_context table stays populated and downstream queries aren't starved of context.
                eer_direction = "above" if row["eer_proxy"] > 0 else "below"
                diagnostic = (
                    f"[PROXY ANNOTATION — API unavailable: {type(api_err).__name__}] "
                    f"{row['manufacturer_name']} recorded a FE2C score of {row['fe2c_efficiency_score']:.1f} "
                    f"({row['z_score']:.2f}z {direction} generation average) at {row['circuit_name']}. "
                    f"EER proxy of {row['eer_proxy']:.4f} and regen index of {row['regen_opportunity_index']:.4f} "
                    f"suggest {'efficient energy conversion relative to the field' if row['z_score'] > 0 else 'energy management challenges under these circuit conditions'}. "
                    f"Weather: {weather_str}. Context source: {source} (similarity {score:.3f})."
                )
                print(f"  WARNING: API call failed for stint {stint_id} — writing proxy annotation.")

            conn.execute(
                "INSERT OR REPLACE INTO rag_context "
                "(stint_id,manufacturer_name,season,race_round,race_name,"
                "qualitative_context,retrieval_score,source_document) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (stint_id, row["manufacturer_name"], int(row["season"]),
                 int(row["race_round"]), row["race_name"], diagnostic, score, source))
            conn.execute(
                "UPDATE race_stints SET rag_context_flag=1 WHERE stint_id=?", (stint_id,))
            conn.commit()
            print(f"  Stint {stint_id}: {diagnostic[:80]}...")
            time.sleep(1.2)


def run_rag_query(question: str) -> None:
    """One-shot RAG answer using ChromaDB + optional structured DB context."""
    import anthropic as _anthropic  # type: ignore[import-untyped]

    collection  = _get_chroma_collection(require_populated=True)
    context, score, source = retrieve_context(collection, question)
    print(f"  Source: '{source}' (similarity: {score:.3f})")

    structured = ""
    if DB_PATH.exists():
        with get_connection() as conn:
            known = ["Mahindra","Porsche","Nissan","Jaguar","Maserati","Lucid","McLaren","ABT"]
            teams = [t for t in known if t.lower() in question.lower()]
            if teams:
                rows = pd.read_sql_query(
                    "SELECT m.manufacturer_name, s.season, s.race_name, "
                    "s.fe2c_efficiency_score, s.regen_opportunity_index, s.eer_proxy "
                    "FROM race_stints s "
                    "JOIN dim_manufacturer m ON s.manufacturer_id=m.manufacturer_id "
                    "WHERE m.manufacturer_name LIKE ? AND s.fe2c_efficiency_score IS NOT NULL "
                    "ORDER BY s.season, s.race_round LIMIT 10",
                    conn, params=(f"%{teams[0]}%",))
                if not rows.empty:
                    structured = "\n\nDB RECORDS:\n" + rows.to_string(index=False)

    answer = _anthropic.Anthropic().messages.create(
        model=CLAUDE_MODEL, max_tokens=400,
        messages=[{"role": "user", "content":
            f"FE2C analyst — Lucid Motors formula E to consumer EV case study.\n"
            f"QUESTION: {question}\nCONTEXT:\n{context}{structured}\n"
            f"Answer in 3-4 sentences. No filler phrases."}]
    ).content[0].text.strip()
    print(f"\nAnswer:\n{answer}\n")


def print_rag_stats() -> None:
    collection = _get_chroma_collection()
    print(f"ChromaDB chunks: {collection.count()}")
    if DB_PATH.exists():
        with get_connection() as conn:
            n_e = conn.execute("SELECT COUNT(*) FROM race_stints WHERE rag_context_flag=1").fetchone()[0]
            n_c = conn.execute("SELECT COUNT(*) FROM rag_context").fetchone()[0]
            print(f"Enriched stints: {n_e}  |  RAG context rows: {n_c}")


# Layer 3: Agent tools
# Each function takes a dict of arguments matching its JSON schema and returns a plain
# string. Errors are returned as strings rather than raised so Claude can read the error
# message, self-correct, and retry with different arguments — this is the core property
# that makes tool-use agents robust on malformed or ambiguous user questions.
#
# Tool design principle: every tool covers a distinct analytical surface. There is no
# overlap between query_race_database (structured SQL) and search_race_documents
# (semantic text). The agent learns to combine them for complete answers — SQL for the
# numbers, ChromaDB for the narrative context behind those numbers.

def tool_query_database(args: dict) -> str:
    """Execute a SELECT query against the FE2C SQLite database."""
    sql = args.get("sql", "").strip()
    if not sql:
        return "ERROR: 'sql' argument is required."
    if not sql.upper().startswith("SELECT"):
        return "ERROR: Only SELECT statements are permitted."
    try:
        with _require_db() as conn:
            df = pd.read_sql_query(sql, conn)
        if df.empty:
            return "Query returned no rows."
        return df.to_string(index=False, max_rows=30)
    except Exception as e:
        return f"SQL ERROR: {e}"


def tool_search_documents(args: dict) -> str:
    """Semantic search over the ChromaDB vector store."""
    query = args.get("query", "").strip()
    if not query:
        return "ERROR: 'query' argument is required."
    try:
        collection = _get_chroma_collection(require_populated=True)
        results    = collection.query(
            query_texts=[query],
            n_results=min(TOP_K, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        if not results["documents"] or not results["documents"][0]:
            return "No relevant documents found."
        docs  = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        sims  = [round(1 - d / 2, 4) for d in dists]
        chunks = [
            f"[Source: {m.get('source','unknown')} | Similarity: {s:.3f}]\n{d}"
            for d, m, s in zip(docs, metas, sims)
        ]
        return "\n\n---\n\n".join(chunks)
    except Exception as e:
        return f"SEARCH ERROR: {e}"


def tool_ca_market_data(args: dict) -> str:
    """Pull California EV registration data, optionally filtered by brand or county."""
    brand  = args.get("brand", "").strip().upper() or None
    county = args.get("county", "").strip() or None
    try:
        conditions: list[str] = [
            "r.registration_count > 0",
            "r.powertrain_type IN ('BEV','PHEV','FCEV')",
        ]
        params: list = []
        if brand:
            conditions.append("r.brand_raw = ?")
            params.append(brand)
        if county:
            conditions.append("r.county = ?")
            params.append(county)
        where = " AND ".join(conditions)
        with _require_db() as conn:
            df = pd.read_sql_query(f"""
                SELECT r.county, r.brand_raw, r.powertrain_type,
                       SUM(r.registration_count) AS total_regs,
                       r.lucid_motors_flag,
                       m.manufacturer_name
                FROM ca_ev_registrations r
                LEFT JOIN dim_manufacturer m ON r.manufacturer_id = m.manufacturer_id
                WHERE {where}
                GROUP BY r.county, r.brand_raw, r.powertrain_type
                ORDER BY r.county, total_regs DESC
            """, conn, params=params)
        if df.empty:
            return "No CA registration data found for those filters."
        county_totals = df.groupby("county")["total_regs"].sum().rename("county_total")
        df = df.merge(county_totals, on="county", how="left")
        df["share_pct"] = (df["total_regs"] / df["county_total"] * 100).round(2)
        lucid_total = int(df[df["lucid_motors_flag"] == 1]["total_regs"].sum())
        out = df[["county", "brand_raw", "powertrain_type", "total_regs", "share_pct"]].to_string(index=False)
        if lucid_total:
            out += f"\n\nLucid total in dataset: {lucid_total:,} registrations"
        return out
    except Exception as e:
        return f"CA DATA ERROR: {e}"


def tool_efficiency_comparison(args: dict) -> str:
    """Compare FE2C efficiency metrics for a team against the field or a specific rival."""
    team_name  = args.get("team_name", "").strip()
    compare_to = args.get("compare_to", "").strip() or None
    gen_filter = args.get("gen_filter", "").strip() or None
    if not team_name:
        return "ERROR: 'team_name' argument is required."
    try:
        params: list = []
        gen_clause   = ""
        if gen_filter:
            gen_clause = "AND g.gen_name = ?"
            params.append(gen_filter)
        with _require_db() as conn:
            df = pd.read_sql_query(
                f"""
                SELECT m.manufacturer_name, g.gen_name,
                       s.fe2c_efficiency_score, s.regen_opportunity_index,
                       s.eer_proxy, s.avg_lap_velocity_kmh,
                       s.positions_gained, s.season
                FROM race_stints s
                JOIN dim_manufacturer m ON s.manufacturer_id = m.manufacturer_id
                JOIN dim_battery_gen  g ON s.gen_id = g.gen_id
                WHERE s.classified = 1
                  AND s.fe2c_efficiency_score IS NOT NULL
                  AND s.regen_opportunity_index IS NOT NULL
                  {gen_clause}
                """,
                conn, params=params)
        if df.empty:
            return "No efficiency data found."

        team_df = df[df["manufacturer_name"].str.contains(team_name, case=False, na=False)]
        if team_df.empty:
            return f"Team '{team_name}' not found. Available: {df['manufacturer_name'].unique().tolist()}"

        if compare_to:
            other_df = df[df["manufacturer_name"].str.contains(compare_to, case=False, na=False)]
            if other_df.empty:
                return f"Comparison team '{compare_to}' not found."
            baseline_df, baseline_label = other_df, compare_to
        else:
            baseline_df    = df[~df["manufacturer_name"].str.contains(team_name, case=False, na=False)]
            baseline_label = "field average"

        def summarize(d: pd.DataFrame, label: str) -> str:
            fe2c  = d["fe2c_efficiency_score"]
            regen = d["regen_opportunity_index"]
            return (
                f"{label}:\n"
                f"  FE2C score:   {fe2c.mean():.2f} (±{fe2c.std():.2f})\n"
                f"  Regen index:  {regen.mean():.6f}\n"
                f"  EER proxy:    {d['eer_proxy'].mean():.4f}\n"
                f"  Avg velocity: {d['avg_lap_velocity_kmh'].mean():.1f} km/h\n"
                f"  Pos gained:   {d['positions_gained'].mean():.2f}\n"
                f"  Stints: {len(d)}  |  Seasons: {d['season'].nunique()}"
            )

        t_stat, p_val = stats.ttest_ind(
            team_df["fe2c_efficiency_score"].dropna(),
            baseline_df["fe2c_efficiency_score"].dropna(),
            equal_var=False)
        regen_delta = (
            (team_df["regen_opportunity_index"].mean() - baseline_df["regen_opportunity_index"].mean())
            / max(baseline_df["regen_opportunity_index"].mean(), 1e-9) * 100
        )
        parts = [
            summarize(team_df, team_name),
            summarize(baseline_df, baseline_label),
            (f"Stats: t={t_stat:.3f}, p={p_val:.4f} "
             f"({'significant' if p_val < 0.05 else 'not significant'} at α=0.05)\n"
             f"Regen delta vs {baseline_label}: {regen_delta:+.1f}%"),
        ]
        if gen_filter:
            parts.append(f"Generation filter: {gen_filter}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"EFFICIENCY ERROR: {e}"


def tool_range_simulation(args: dict) -> str:
    """Run the FE2C range transfer simulation for the Lucid Air."""
    gen_filter = args.get("gen_filter", "Gen3").strip()
    try:
        with _require_db() as conn:
            df = pd.read_sql_query(
                """
                SELECT s.regen_opportunity_index, s.fe2c_efficiency_score,
                       m.manufacturer_id, m.manufacturer_name
                FROM race_stints s
                JOIN dim_manufacturer m ON s.manufacturer_id = m.manufacturer_id
                JOIN dim_battery_gen  g ON s.gen_id = g.gen_id
                WHERE s.classified = 1
                  AND s.regen_opportunity_index IS NOT NULL
                  AND s.fe2c_efficiency_score IS NOT NULL
                  AND g.gen_name = ?
                """,
                conn, params=[gen_filter])
        if df.empty:
            return f"No {gen_filter} data found."

        lucid_mask  = df["manufacturer_id"].isin([LUCID_MFR_ID, MAHINDRA_MFR_ID])
        lucid_regen = df.loc[lucid_mask,  "regen_opportunity_index"].values
        field_regen = df.loc[~lucid_mask, "regen_opportunity_index"].values

        if len(lucid_regen) == 0:
            # No Lucid/Mahindra telemetry in this generation fall back to the
            # engineered proxy recovery ratio from Technical Report §8 so the agent
            # can still return a directional estimate rather than an error.
            proxy = ENGINEERED_PROXIES["gen3_recovery_ratio"]
            baseline = ENGINEERED_PROXIES["gen2_regen_efficiency"]
            transfer = max((proxy - baseline) / max(baseline, 1e-9), 0.0) * REGEN_WEIGHT
            sim = LUCID_AIR_EPA_RANGE * (1 + transfer)
            return (
                f"[PROXY ESTIMATE — no Lucid/Mahindra stints in {gen_filter}]\n"
                f"Engineered proxy (Technical Report §8):\n"
                f"  Gen3 recovery ratio: {proxy:.0%}  |  Gen2 baseline: {baseline:.0%}\n"
                f"  Estimated transfer factor: {transfer:.4f}\n"
                f"  Lucid Air EPA baseline: {LUCID_AIR_EPA_RANGE} mi\n"
                f"  Proxy-estimated range: {sim:.1f} mi\n"
                f"NOTE: Theoretical estimate only. Run --reset then --ingest for data-driven results."
            )
        if len(field_regen) == 0:
            return "No field stints for comparison."

        lucid_mean      = lucid_regen.mean()
        field_mean      = field_regen.mean()
        t_stat, p_val   = stats.ttest_ind(lucid_regen, field_regen, equal_var=False)
        transfer_factor = float(np.clip((lucid_mean - field_mean) / max(field_mean, 1e-9), 0, 2.0))
        sim_range       = LUCID_AIR_EPA_RANGE * (1 + transfer_factor * REGEN_WEIGHT)

        np.random.seed(42)
        boot_idx     = np.random.choice(len(field_regen), size=(1000, len(field_regen)), replace=True)
        boot_means   = field_regen[boot_idx].mean(axis=1)
        boot_factors = np.clip((lucid_mean - boot_means) / np.maximum(boot_means, 1e-9), 0, 2.0)
        boot_ranges  = LUCID_AIR_EPA_RANGE * (1 + boot_factors * REGEN_WEIGHT)
        ci_lo = float(np.percentile(boot_ranges, 2.5))
        ci_hi = float(np.percentile(boot_ranges, 97.5))

        lucid_fe2c = df.loc[lucid_mask,  "fe2c_efficiency_score"].mean()
        field_fe2c = df.loc[~lucid_mask, "fe2c_efficiency_score"].mean()
        teams      = df.loc[lucid_mask, "manufacturer_name"].unique().tolist()

        return (
            f"FE2C Range Simulation ({gen_filter})\n\n"
            f"Lucid chain teams: {teams}\n"
            f"Lucid stints: {len(lucid_regen)}  |  Field stints: {len(field_regen)}\n\n"
            f"Lucid avg regen:  {lucid_mean:.6f}\n"
            f"Field avg regen:  {field_mean:.6f}\n"
            f"Transfer factor:  {transfer_factor:.4f} ({transfer_factor * 100:.1f}% regen advantage)\n"
            f"Regen weight:     {REGEN_WEIGHT} (15% — conservative real-world regen contribution)\n\n"
            f"Lucid Air EPA baseline:  {LUCID_AIR_EPA_RANGE} mi\n"
            f"Simulated range:         {sim_range:.1f} mi\n"
            f"95% Bootstrap CI:        {ci_lo:.1f} – {ci_hi:.1f} mi\n\n"
            f"t={t_stat:.3f}, p={p_val:.4f} "
            f"({'significant' if p_val < 0.05 else 'not significant'} at α=0.05)\n\n"
            f"FE2C score — Lucid chain: {lucid_fe2c:.2f}  |  Field: {field_fe2c:.2f}\n\n"
            f"NOTE: regen_opportunity_index is a proxy (lap variance + elevation), "
            f"not live telemetry. Treat simulated range as directional."
        )
    except Exception as e:
        return f"SIMULATION ERROR: {e}"


def tool_new_circuit_prediction(args: dict) -> str:
    """Cold-start prediction for circuits with no historical FE2C data (§5.3)."""
    surface_type    = args.get("surface_type",   "street").strip().lower()
    traction_demand = args.get("traction_demand", "medium").strip().lower()
    try:
        elevation_delta = float(args.get("elevation_delta_m", 10.0))
    except (TypeError, ValueError):
        return "ERROR: elevation_delta_m must be a numeric value."

    valid_surfaces = {"street", "permanent", "hybrid"}
    valid_traction = {"low", "medium", "high"}
    if surface_type not in valid_surfaces:
        return f"ERROR: surface_type must be one of {sorted(valid_surfaces)}."
    if traction_demand not in valid_traction:
        return f"ERROR: traction_demand must be one of {sorted(valid_traction)}."

    try:
        with _require_db() as conn:
            #Rank circuits by elevation proximity, then filter by at least one matching
            #categorical feature so the prior isn't drawn from unrelated venues.
            df = pd.read_sql_query(
                """
                WITH ranked_circuits AS (
                    SELECT t.track_id, t.circuit_name, t.surface_type,
                           t.traction_demand, t.elevation_delta_m,
                           ABS(t.elevation_delta_m - ?) AS elev_diff
                    FROM dim_track t
                    WHERE t.surface_type = ? OR t.traction_demand = ?
                    ORDER BY elev_diff
                    LIMIT 5
                )
                SELECT rc.circuit_name, rc.surface_type, rc.traction_demand,
                       rc.elevation_delta_m, rc.elev_diff,
                       AVG(s.fe2c_efficiency_score)                        AS avg_fe2c_score,
                       AVG(s.regen_opportunity_index)                      AS avg_regen_index,
                       AVG(CAST(s.positions_gained AS REAL) / NULLIF(s.eer_proxy, 0)) AS avg_delta_e,
                       COUNT(s.stint_id)                                   AS n_stints
                FROM ranked_circuits rc
                JOIN race_stints s ON rc.track_id = s.track_id
                WHERE s.classified = 1
                  AND s.fe2c_efficiency_score IS NOT NULL
                GROUP BY rc.circuit_name, rc.surface_type,
                         rc.traction_demand, rc.elevation_delta_m, rc.elev_diff
                ORDER BY rc.elev_diff
                """,
                conn, params=[elevation_delta, surface_type, traction_demand])

        if df.empty:
            return (
                f"No similar circuits found for surface_type='{surface_type}', "
                f"traction_demand='{traction_demand}'. "
                "Try broadening one of the categorical filters."
            )

        pred_fe2c  = df["avg_fe2c_score"].mean()
        pred_regen = df["avg_regen_index"].mean()
        pred_delta = df["avg_delta_e"].mean()

        lines = [
            "New Circuit Efficiency Prediction (Cold-Start Prior)",
            f"Input: surface={surface_type}, elevation={elevation_delta}m, traction={traction_demand}",
            "",
            f"Predicted FE2C score:   {pred_fe2c:.2f}  (avg across {len(df)} analogous circuits)",
            f"Predicted regen index:  {pred_regen:.6f}",
            f"Predicted delta_e:      {pred_delta:.6f}  "
            f"({'efficient' if pred_delta < 0 else 'energy-costly'} vs race average)",
            "",
            "Analogue circuits used as priors:",
            df[["circuit_name","surface_type","elevation_delta_m",
                "avg_fe2c_score","avg_regen_index","n_stints"]].to_string(index=False),
            "\nNOTE: Confidence scales with how closely the analogues match.",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"PREDICTION ERROR: {e}"


def tool_calculate_regen_index(args: dict) -> str:
    """
    Implements Technical Report Section 2: rank team regen performance by computing
    energy recovered via regeneration vs. total energy density available per generation.

    The formula treats regen_opportunity_index (our proxy for recovered energy) as a
    fraction of the declared battery_kwh capacity — giving a normalized efficiency ratio
    that is comparable across Gen2 and Gen3 despite different pack sizes.

    Regen Index = mean(regen_opportunity_index) / battery_kwh
    Ranked within generation cohort and expressed as a percentage of the top performer.

    Also compares the Gen3 field efficiency curve against Lucid consumer benchmarks
    to answer: how does racing regen discipline translate to real-world range?
    """
    gen_filter = args.get("gen_filter", "").strip() or None
    top_n      = int(args.get("top_n", 8))

    try:
        gen_clause = "AND g.gen_name = ?" if gen_filter else ""
        params: list = [gen_filter] if gen_filter else []

        with _require_db() as conn:
            df = pd.read_sql_query(
                f"""
                SELECT m.manufacturer_name,
                       g.gen_name,
                       g.battery_kwh,
                       AVG(s.regen_opportunity_index) AS avg_regen_proxy,
                       AVG(s.fe2c_efficiency_score)   AS avg_fe2c_score,
                       COUNT(s.stint_id)              AS n_stints
                FROM race_stints s
                JOIN dim_manufacturer m ON s.manufacturer_id = m.manufacturer_id
                JOIN dim_battery_gen  g ON s.gen_id = g.gen_id
                WHERE s.classified = 1
                  AND s.regen_opportunity_index IS NOT NULL
                  {gen_clause}
                GROUP BY m.manufacturer_name, g.gen_name, g.battery_kwh
                HAVING n_stints >= 5
                ORDER BY g.gen_name, avg_regen_proxy DESC
                """,
                conn, params=params,
            )

        if df.empty:
            # No database records available return engineered proxy estimates from
            # Technical Report §8 so the agent can still discuss regen performance
            # directionally without hard-failing on a missing pipeline run.
            g3 = ENGINEERED_PROXIES["gen3_recovery_ratio"]
            g2 = ENGINEERED_PROXIES["gen2_regen_efficiency"]
            return (
                "[PROXY ESTIMATE — no regen data in database]\n"
                "Engineered proxy values (Technical Report §8 / Page 7):\n"
                f"  Gen3 assumed recovery ratio: {g3:.0%} "
                f"(front {ENGINEERED_PROXIES['gen3_front_regen_kw']:.0f} kW + "
                f"rear {ENGINEERED_PROXIES['gen3_rear_regen_kw']:.0f} kW)\n"
                f"  Gen2 regen efficiency:       {g2:.0%} (single rear motor)\n"
                f"  Gen3 improvement over Gen2:  +{(g3-g2)/g2*100:.0f}%\n"
                "\nRun the full pipeline (--reset) to replace these with race-derived figures."
            )

        #Normalize within each gen cohort so Gen2 and Gen3 are ranked independently.
        #Dividing by battery_kwh collapses the pack-size advantage a team on a 38.5 kWh
        #Gen3 pack that matches a 52 kWh Gen2 team's raw index is genuinely more efficient.
        df["regen_index_ratio"] = df["avg_regen_proxy"] / df["battery_kwh"].clip(lower=1e-6)

        df["gen_rank"] = (
            df.groupby("gen_name")["regen_index_ratio"]
            .rank(ascending=False)
            .astype(int)
        )
        gen_top = df.groupby("gen_name")["regen_index_ratio"].transform("max").clip(lower=1e-9)
        df["pct_of_leader"] = (df["regen_index_ratio"] / gen_top * 100).round(1)

        lines = ["FE2C Regen Index — Section 2 Implementation", ""]
        lines.append(f"Formula: regen_index_ratio = avg_regen_proxy / battery_kwh")
        lines.append(f"Normalized within generation cohort. Higher = more energy recovered per kWh available.")
        lines.append("")

        for gen_name, grp in df.groupby("gen_name"):
            pack_kwh = grp["battery_kwh"].iloc[0]
            lines.append(f"[ {gen_name} | Pack: {pack_kwh} kWh ]")
            lines.append(f"  {'Rank':<5} {'Manufacturer':<22} {'Regen Ratio':>12} {'% of Leader':>12} {'FE2C Score':>11} {'Stints':>7}")
            for _, row in grp.nsmallest(top_n, "gen_rank").iterrows():
                lucid_tag = " [LUCID CHAIN]" if "mahindra" in row["manufacturer_name"].lower() or "lucid" in row["manufacturer_name"].lower() else ""
                lines.append(
                    f"  {int(row['gen_rank']):<5} {row['manufacturer_name']:<22}"
                    f" {row['regen_index_ratio']:>12.6f} {row['pct_of_leader']:>11.1f}%"
                    f" {row['avg_fe2c_score']:>11.2f} {int(row['n_stints']):>7}{lucid_tag}"
                )
            lines.append("")

        #Consumer benchmark crosswalk: translate the Gen3 regen ratio to estimated
        #range uplift on Lucid's consumer vehicles using the same transfer logic as
        #compute_range_simulation, but expressed per vehicle model.
        gen3_rows = df[df["gen_name"] == "Gen3"]
        if not gen3_rows.empty:
            lucid_chain  = gen3_rows[gen3_rows["manufacturer_name"].str.contains("Mahindra|Lucid", case=False, na=False)]
            field_rows   = gen3_rows[~gen3_rows["manufacturer_name"].str.contains("Mahindra|Lucid", case=False, na=False)]
            lucid_ratio  = lucid_chain["regen_index_ratio"].mean() if not lucid_chain.empty else 0.0
            field_ratio  = field_rows["regen_index_ratio"].mean() if not field_rows.empty else 0.0
            transfer_pct = float(np.clip((lucid_ratio - field_ratio) / max(field_ratio, 1e-9), 0, 2.0)) * REGEN_WEIGHT * 100

            lines.append("Gen3 Efficiency Curve vs. Lucid Consumer Benchmarks")
            lines.append(f"  Lucid/Mahindra regen ratio advantage: {lucid_ratio:.6f} vs field {field_ratio:.6f}")
            lines.append(f"  Estimated real-world range uplift (15% regen weight): +{transfer_pct:.1f}%")
            lines.append("")
            lines.append(f"  {'Model':<20} {'EPA Range':>10} {'Simulated Range':>16} {'Battery kWh':>12} {'Segment'}")
            for model_name, spec in LUCID_BENCHMARKS.items():
                sim_range = spec["epa_range_mi"] * (1 + transfer_pct / 100)
                lines.append(
                    f"  {model_name:<20} {spec['epa_range_mi']:>9} mi"
                    f" {sim_range:>14.1f} mi {spec['battery_kwh']:>11.1f} kWh  {spec['segment']}"
                )
            lines.append("")
            lines.append("  NOTE: Simulated range is directional. regen_opportunity_index is a proxy,")
            lines.append("  not live telemetry. The 15% regen weight is a conservative assumption per §6.3.")

        return "\n".join(lines)

    except Exception as e:
        return f"REGEN INDEX ERROR: {e}"


def tool_strategy_recommendation(args: dict) -> str:
    """ZIP/county-weighted density-vs-recovery strategy recommendation (§6.3)."""
    county = args.get("county", "").strip()
    #Threshold calibrated to the CA dataset: LA/Orange exceed this; Kern/Stanislaus sit below.
    DENSITY_THRESHOLD = 500

    try:
        params: list = ["BEV", "PHEV", "FCEV"]
        county_clause = ""
        if county:
            county_clause = "AND r.county = ?"
            params.append(county)

        with _require_db() as conn:
            df = pd.read_sql_query(
                f"""
                SELECT r.county, r.zip_code,
                       SUM(r.registration_count) AS zip_total_regs,
                       MAX(r.lucid_motors_flag)  AS has_lucid
                FROM ca_ev_registrations r
                WHERE r.powertrain_type IN (?, ?, ?)
                  AND r.registration_count > 0
                  {county_clause}
                GROUP BY r.county, r.zip_code
                ORDER BY r.county, zip_total_regs DESC
                """,
                conn, params=params)

        if df.empty:
            return "No CA registration data found. Run the full pipeline first."

        county_summary = (
            df.groupby("county")
            .agg(
                total_evs      =("zip_total_regs", "sum"),
                median_zip_evs =("zip_total_regs", "median"),
                lucid_present  =("has_lucid",      "max"),
            ).reset_index()
        )

        lines = [
            "FE2C Strategy Recommendation — Battery Density vs Recovery Efficiency",
            f"Density threshold (EVs per ZIP): {DENSITY_THRESHOLD:,}",
            "",
            f"{'County':<26} {'Total EVs':>10} {'Median/ZIP':>11} {'Recommendation':<26} Reasoning",
            "-" * 95,
        ]
        for _, row in county_summary.iterrows():
            if row["median_zip_evs"] >= DENSITY_THRESHOLD:
                rec    = "Recovery Efficiency"
                reason = "Dense urban stop-and-go — frequent regen cycles (Gen 3 model)"
            else:
                rec    = "Battery Density"
                reason = "Low-infra / long-haul — range buffer outweighs regen gain"
            lucid_tag = "  [LUCID PRESENT]" if row["lucid_present"] else ""
            lines.append(
                f"{row['county']:<26} {int(row['total_evs']):>10,} "
                f"{int(row['median_zip_evs']):>11,}  {rec:<26} {reason}{lucid_tag}"
            )
        lines += [
            "",
            "Framework (per FE2C Technical Report §6.3):",
            "  Recovery Efficiency — optimised regen, lighter pack, dynamic charging.",
            "                        Aligns with Formula E Gen 3 design philosophy.",
            "  Battery Density     — larger pack, higher upfront cost, longer range buffer.",
            "                        Lucid Air Grand Touring as benchmark.",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"STRATEGY ERROR: {e}"


# Agent: tool registry and system prompt
# TOOLS maps each tool name to (function, Anthropic JSON schema). The schema is what
# Claude sees, it determines when Claude decides to call each tool and with what arguments.
# Schema descriptions are written to bias Claude toward the right tool for each question
# class: regen questions → calculate_regen_index first; market questions → ca_market_data
# AND strategy_recommendation together; cold-start venues → predict_new_circuit_efficiency.
#
# The SYSTEM_PROMPT establishes the "Live Laboratory" persona from the Technical Report
# Claude answers as a Technical Lead who views every race result as a data point in a
# multi-season consumer EV battery study, not as a sports commentator.

TOOLS: dict[str, tuple[Any, dict]] = {
    "calculate_regen_index": (
        tool_calculate_regen_index,
        {
            "name": "calculate_regen_index",
            "description": (
                "Implements Technical Report Section 2: rank all teams by the ratio of "
                "energy recovered via regenerative braking vs. total energy density available "
                "(regen_proxy / battery_kwh), normalized within generation cohort. "
                "Also crosswalks the Gen3 efficiency curve against Lucid consumer vehicle "
                "benchmarks (Air Sapphire, Air Grand Touring, Gravity) to answer: "
                "'How does the Gen3 efficiency curve compare to a Lucid Air Sapphire?' "
                "Use this tool whenever a question involves regen ranking, energy recovery "
                "efficiency, or comparing racing performance to Lucid consumer specs."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "gen_filter": {
                        "type": "string",
                        "description": "'Gen2', 'Gen3', or leave empty for both generations.",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of teams to show per generation (default 8).",
                    },
                },
                "required": [],
            },
        },
    ),
    "query_race_database": (
        tool_query_database,
        {
            "name": "query_race_database",
            "description": (
                "Execute a SELECT query against the FE2C SQLite database. "
                "Use for structured questions about race performance, efficiency scores, "
                "regen indexes, season statistics, and manufacturer comparisons. "
                "Tables: race_stints, dim_manufacturer, dim_battery_gen, dim_track, "
                "dim_weather, ca_ev_registrations, rag_context. "
                "Always JOIN dim_manufacturer to get manufacturer names. "
                "Filter classified=1 for valid race finishes."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": (
                            "A valid SQLite SELECT statement. Example: "
                            "SELECT m.manufacturer_name, "
                            "AVG(s.fe2c_efficiency_score) as avg_score "
                            "FROM race_stints s "
                            "JOIN dim_manufacturer m "
                            "ON s.manufacturer_id = m.manufacturer_id "
                            "WHERE s.classified = 1 AND s.gen_id = 3 "
                            "GROUP BY m.manufacturer_name ORDER BY avg_score DESC"
                        ),
                    }
                },
                "required": ["sql"],
            },
        },
    ),
    "search_race_documents": (
        tool_search_documents,
        {
            "name": "search_race_documents",
            "description": (
                "Semantic search over the ChromaDB vector store containing Formula E "
                "season summaries, Lucid Motors technology details, Mahindra Racing "
                "history, and Gen3 car specifications from Wikipedia. "
                "Use for qualitative context not in the structured database. "
                "Always combine with query_race_database for complete answers."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query.",
                    }
                },
                "required": ["query"],
            },
        },
    ),
    "get_ca_market_data": (
        tool_ca_market_data,
        {
            "name": "get_ca_market_data",
            "description": (
                "Retrieve California EV registration data aggregated by brand and county. "
                "Returns registration counts and market share percentages by county."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "brand":  {"type": "string", "description": "Brand filter e.g. 'LUCID', 'PORSCHE'. Leave empty for all."},
                    "county": {"type": "string", "description": "County filter e.g. 'Los Angeles'. Leave empty for all."},
                },
                "required": [],
            },
        },
    ),
    "compute_efficiency_comparison": (
        tool_efficiency_comparison,
        {
            "name": "compute_efficiency_comparison",
            "description": (
                "Compare FE2C efficiency metrics for a named team against the field "
                "average or a specific rival. Returns FE2C score, regen index, EER proxy, "
                "velocity, positions gained, and Welch's t-test significance. "
                "Use gen_filter='Gen3' to isolate Lucid technology seasons."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "team_name":  {"type": "string", "description": "Manufacturer name e.g. 'Mahindra'."},
                    "compare_to": {"type": "string", "description": "Rival to compare against. Leave empty for field average."},
                    "gen_filter": {"type": "string", "description": "'Gen2' or 'Gen3'. Leave empty for all."},
                },
                "required": ["team_name"],
            },
        },
    ),
    "get_range_simulation": (
        tool_range_simulation,
        {
            "name": "get_range_simulation",
            "description": (
                "Bootstrapped Lucid Air range simulation: models how Lucid/Mahindra's "
                "Formula E regen performance translates to estimated real-world range. "
                "Returns simulated range with 95% CI."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "gen_filter": {"type": "string", "description": "'Gen3' (default) or 'Gen2'."},
                },
                "required": [],
            },
        },
    ),
    "predict_new_circuit_efficiency": (
        tool_new_circuit_prediction,
        {
            "name": "predict_new_circuit_efficiency",
            "description": (
                "Cold-start efficiency prediction for a circuit with no Formula E history. "
                "Finds analogue circuits and returns aggregate FE2C score and regen index as a prior."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "surface_type":      {"type": "string", "description": "'street', 'permanent', or 'hybrid'."},
                    "elevation_delta_m": {"type": "number", "description": "Elevation change across the circuit in metres."},
                    "traction_demand":   {"type": "string", "description": "'low', 'medium', or 'high'."},
                },
                "required": [],
            },
        },
    ),
    "get_strategy_recommendation": (
        tool_strategy_recommendation,
        {
            "name": "get_strategy_recommendation",
            "description": (
                "Returns a county-level battery strategy recommendation for California: "
                "Battery Density vs Recovery Efficiency, driven by EV registration density per ZIP."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "county": {"type": "string", "description": "Filter to a single county e.g. 'Los Angeles'. Leave empty for all."},
                },
                "required": [],
            },
        },
    ),
}

TOOL_SCHEMAS = [schema for _, schema in TOOLS.values()]

SYSTEM_PROMPT = """You are the FE2C Technical Lead the senior analyst for the Formula E to Consumer
(FE2C) project. Your core thesis: Formula E is not a race series. It is a live laboratory
where competitive pressure compresses battery R&D cycles that would otherwise take years in
consumer contexts.

Every race is a controlled experiment in energy management under hard constraints. Every stint
record in this database is a data point in a multi-season study of how teams convert stored
electrochemical energy into mechanical advantage — and what that discipline teaches us about
building better consumer EVs.

Lucid Motors is the direct line of transfer. As powertrain supplier to the Mahindra Formula E
team, Lucid occupies a rare position: the same engineering team that optimizes regen recovery
and thermal management under race conditions also designs the Air Sapphire's 300 Wh/kg cell
architecture. This project quantifies that transfer.

You have eight tools:
  calculate_regen_index          — Section 2 implementation: regen ratio ranked by gen cohort,
                                   crosswalked against Lucid Air Sapphire / Grand Touring benchmarks
  query_race_database            — structured SQL queries (delta_e proxy: negative = efficient)
  search_race_documents          — semantic search over Wikipedia / qualitative context
                                   (prioritize Gen2 vs Gen3 battery innovation themes)
  get_ca_market_data             — California EV registration data by brand and county
  compute_efficiency_comparison  — FE2C score + regen index comparison with Welch's t-test
  get_range_simulation           — bootstrapped Lucid Air range transfer simulation
  predict_new_circuit_efficiency — cold-start prediction for venues with no FE history (§5.3)
  get_strategy_recommendation    — density vs recovery recommendation by CA county (§6.3)

Operating rules:
- Always use at least one tool before answering. Never guess from memory.
- For efficiency or regen questions: use calculate_regen_index first, then query_race_database.
- For technology transfer questions: search_race_documents with explicit Gen2 vs Gen3 framing.
- For Lucid consumer comparisons: calculate_regen_index surfaces the benchmark crosswalk directly.
- For market strategy: get_strategy_recommendation AND get_ca_market_data together.
- For cold-start circuits: predict_new_circuit_efficiency.
- Cite specific numbers from tool outputs. Vague answers are not acceptable.
- If a tool returns an error or empty result, report the engineered proxy assumption
  (per Technical Report §8 data honesty framework) rather than refusing to answer.
- Gen3 seasons (9 and 10) are the primary evidence for the FE2C thesis. Lucid supplied
  powertrain technology to Mahindra Racing in those seasons.
- delta_e proxy interpretation: negative = team consumed less energy than race average
  (genuine efficiency signal); positive = energy-costly performance. Always state the sign.
- The regen_opportunity_index is a proxy derived from lap variance and elevation change —
  not live telemetry. Always acknowledge this limitation in final answers.
- When comparing Gen3 racing to Lucid consumer specs, use the benchmark crosswalk from
  calculate_regen_index: Air Sapphire (687 mi EPA, 300 Wh/kg), Grand Touring (516 mi EPA)."""


# Agent: preflight and the tool-use loop
# agent_preflight() fails fast before any API call is made missing DB or empty ChromaDB
# means every tool call would return an error anyway. Better to surface the fix command
# immediately than to let the user burn API tokens on a broken session.
#
# run_agent() implements the standard Anthropic tool-use pattern:
#   1. Send user message + tool schemas to Claude
#   2. Claude responds with text and/or tool_use blocks
#   3. Execute each tool call, collect results as tool_result blocks
#   4. Append both assistant response and tool results to message history
#   5. Repeat until Claude returns stop_reason="end_turn" (no more tool calls)
#
# Prompt caching on the system prompt and last tool schema reduces token cost by ~40%
# on multi-step questions where the same static context is sent on every iteration.
# MAX_ITERATIONS = 8 is a safety cap most questions resolve in 2-4 iterations.

def agent_preflight() -> None:
    """Fail fast before the agent starts — surface missing prerequisites clearly."""
    py       = sys.executable
    script   = Path(__file__)
    errors   = []

    if not DB_PATH.exists():
        errors.append(
            f"Database missing: {DB_PATH}\n  Fix: {py} {script} --reset")
    else:
        try:
            with get_connection() as conn:
                n = conn.execute(
                    "SELECT COUNT(*) FROM race_stints "
                    "WHERE fe2c_efficiency_score IS NOT NULL").fetchone()[0]
                if n == 0:
                    errors.append(
                        f"fe2c_efficiency_score not computed.\n  Fix: {py} {script} --reset")
        except Exception as e:
            errors.append(f"Database error: {e}")

    if not CHROMA_DIR.exists():
        errors.append(
            f"ChromaDB missing: {CHROMA_DIR}\n  Fix: {py} {script} --ingest")

    if errors:
        print("PREFLIGHT FAILED:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    print("Preflight OK.\n")


def run_agent(question: str, verbose: bool = True) -> str:
    """
    Run the agentic loop for a single question.
    Returns the agent's final answer as a string.
    """
    import anthropic as _anthropic  # type: ignore[import-untyped]

    client   = _anthropic.Anthropic()
    messages = [{"role": "user", "content": question}]

    #Cache the system prompt and tool list — both are static across all iterations.
    #Cache hits avoid re-encoding ~2k tokens per turn, compounding across multi-step questions.
    cached_system = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
    cached_tools  = TOOL_SCHEMAS[:-1] + [
        {**TOOL_SCHEMAS[-1], "cache_control": {"type": "ephemeral"}}
    ]

    for iteration in range(MAX_ITERATIONS):
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2000,
            system=cached_system,
            tools=cached_tools,
            messages=messages,
        )

        if verbose:
            print(f"\n[Iteration {iteration + 1}] stop_reason={response.stop_reason}")

        text_blocks = [b for b in response.content if b.type == "text"]
        tool_blocks = [b for b in response.content if b.type == "tool_use"]

        if text_blocks and verbose:
            for tb in text_blocks:
                print(f"  Claude: {tb.text[:200]}{'...' if len(tb.text) > 200 else ''}")

        if response.stop_reason == "end_turn" or not tool_blocks:
            final = " ".join(b.text for b in text_blocks if b.type == "text")
            return final.strip() if final.strip() else "No answer generated."

        tool_results = []
        for tool_call in tool_blocks:
            tool_name = tool_call.name
            tool_args = tool_call.input

            if verbose:
                print(f"  Tool: {tool_name}({json.dumps(tool_args)[:120]})")

            if tool_name not in TOOLS:
                result_text = f"ERROR: Unknown tool '{tool_name}'."
            else:
                tool_fn, _ = TOOLS[tool_name]
                result_text = tool_fn(tool_args)

            if verbose:
                preview = result_text[:300].replace("\n", " ")
                print(f"  Result: {preview}{'...' if len(result_text) > 300 else ''}")

            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": tool_call.id,
                "content":     result_text,
            })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user",      "content": tool_results})

    return "Max iterations reached. Partial analysis may be incomplete."


DEMO_QUESTIONS = [
    (
        "Which manufacturer had the highest average FE2C efficiency score in Gen3, "
        "and how does Lucid/Mahindra's regen index compare to the rest of the field?"
    ),
    (
        "How does the Gen3 efficiency curve compare to a Lucid Air Sapphire? "
        "Use the regen index analysis to show what the racing data implies for "
        "real-world range on Lucid's consumer lineup."
    ),
    (
        "Compare Mahindra's efficiency in Gen3 directly against Porsche. "
        "Which team leads on the regen index, and what does the qualitative "
        "Formula E documentation say about their respective battery strategies?"
    ),
    (
        "Which California counties show the highest Lucid Motors registrations, "
        "and how does Lucid's FE2C efficiency rank among Gen3 competitors?"
    ),
    (
        "What does Formula E's generational battery evolution from Gen2 to Gen3 tell us "
        "about the current consumer EV landscape, and where does Lucid sit in that progression?"
    ),
]


# Full pipeline runner orchestrates all seven ETL and analytics stages
# Stage order matters: dimensions must exist before facts, facts before metrics,
# metrics before Week 3 analytics. _migrate_db() runs after init_db() every time
# so existing databases gain new columns (delta_e) without requiring --reset.

def run_pipeline(skip_weather: bool = False, reset: bool = False, save_csv: bool = True) -> int:
    print("FE2C Optimization Engine — Full Pipeline (Weeks 1 + 2 + 3)")

    print("\n[1/7] Initializing database...")
    init_db(reset=reset)
    _migrate_db()

    print("\n[2/7] Seeding dimensions...")
    seed_dimensions()

    print("\n[3/7] Formula E race data ETL...")
    with get_connection() as conn:
        run_formula_e_ingestion(conn)

    if skip_weather:
        print("\n[4/7] Weather skipped (--skip-weather)")
    else:
        print("\n[4/7] Weather enrichment...")
        with get_connection() as conn:
            run_weather_ingestion(conn)

    print("\n[5/7] California EV registrations ETL...")
    with get_connection() as conn:
        run_california_ingestion(conn)

    print("\n[6/7] Efficiency metrics (Week 2)...")
    with get_connection() as conn:
        compute_efficiency_metrics(conn)

    print("\n[7/7] Data quality checks...")
    with get_connection() as conn:
        summary = run_quality_checks(conn)

    print("\nWeek 3 — CA Consumer Market Simulation")
    with get_connection() as conn:
        run_week3(conn, save_csv=save_csv)

    print(f"\nPipeline complete. DB: {DB_PATH}")
    return 0 if summary["failed"] == 0 else 1


# Entry point argument routing for pipeline, RAG, and agent modes
# The three mode groups (agent, RAG, pipeline) are mutually exclusive by convention
# not enforced, but the execution order dependency means running --chat before --reset
# will fail at preflight anyway. The --quiet flag is shared across all agent modes.

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FE2C Optimization Engine + Intelligence Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    #Pipeline flags
    parser.add_argument("--skip-weather", action="store_true", help="Skip Open-Meteo API calls")
    parser.add_argument("--reset",        action="store_true", help="Wipe and rebuild database")
    parser.add_argument("--no-csv",       action="store_true", help="Skip Week 3 CSV output")

    #RAG flags
    parser.add_argument("--ingest",  action="store_true", help="Build ChromaDB vector store")
    parser.add_argument("--enrich",  action="store_true", help="RAG enrichment on outlier stints")
    parser.add_argument("--query",   type=str,            help="Ad-hoc natural language query (pipeline RAG)")
    parser.add_argument("--stats",   action="store_true", help="Print RAG and DB stats")

    #Agent flags
    parser.add_argument("--chat",    action="store_true", help="Interactive agent chat loop")
    parser.add_argument("--ask",     type=str,            help="Single agent question, then exit")
    parser.add_argument("--demo",    action="store_true", help="Run built-in demo questions")
    parser.add_argument("--quiet",   action="store_true", help="Suppress iteration output (agent modes)")

    args    = parser.parse_args()
    verbose = not args.quiet

    #Agent modes
    if args.chat or args.ask or args.demo:
        print("FE2C Intelligence Agent")
        print(f"Model: {CLAUDE_MODEL}")
        print(f"DB:    {DB_PATH}")
        print(f"Tools ({len(TOOLS)}): {', '.join(TOOLS.keys())}\n")
        agent_preflight()

        if args.ask:
            answer = run_agent(args.ask, verbose=verbose)
            print(f"\nFINAL ANSWER:\n{answer}")

        elif args.demo:
            print(f"Running {len(DEMO_QUESTIONS)} demo questions...\n")
            for i, q in enumerate(DEMO_QUESTIONS, 1):
                print(f"\nDEMO {i}/{len(DEMO_QUESTIONS)}")
                print(f"Question: {q}")
                print("-" * 60)
                answer = run_agent(q, verbose=verbose)
                print(f"\nFINAL ANSWER:\n{answer}\n")
                print("=" * 60)

        else:
            print("Interactive mode. Type your question and press Enter.")
            print("Type 'exit' to quit.\n")
            while True:
                try:
                    question = input("Question: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting.")
                    break
                if question.lower() in {"exit", "quit", "q", ""}:
                    break
                answer = run_agent(question, verbose=verbose)
                print(f"\nFINAL ANSWER:\n{answer}\n")
        return

    #RAG modes (no agent loop)
    if any([args.ingest, args.enrich, args.query, args.stats]):
        if args.ingest: run_ingest()
        if args.enrich: run_enrichment()
        if args.query:  run_rag_query(args.query)
        if args.stats:  print_rag_stats()
        return

    #Default: full pipeline
    sys.exit(run_pipeline(
        skip_weather=args.skip_weather,
        reset=args.reset,
        save_csv=not args.no_csv,
    ))


if __name__ == "__main__":
    main()
