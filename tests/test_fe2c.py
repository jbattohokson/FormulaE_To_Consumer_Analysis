"""Offline tests: pipeline build, simulation consistency, metric direction, range CI.

Run: python -m pytest tests/
No network or API key needed: the DMV download is forced to fail (synthetic fallback)
and weather is skipped.
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fe2c  # noqa: E402


@pytest.fixture(scope="module")
def built_db(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fe2c")
    mp = pytest.MonkeyPatch()
    mp.setattr(fe2c, "DB_PATH", tmp / "db" / "fe2c.db")
    mp.setattr(fe2c, "RAW_DIR", tmp / "raw")
    mp.setattr(fe2c, "OUT_DIR", tmp / "out")
    (tmp / "db").mkdir()
    (tmp / "raw").mkdir()

    def offline(*a, **k):
        raise fe2c.requests.ConnectionError("offline test")
    mp.setattr(fe2c.requests, "get", offline)
    rc = fe2c.run_pipeline(skip_weather=True, reset=True, save_csv=True)
    yield fe2c.DB_PATH, rc
    mp.undo()


def stints(db):
    with sqlite3.connect(db) as con:
        return pd.read_sql("SELECT * FROM race_stints", con)


def test_pipeline_builds_and_all_quality_checks_pass(built_db):
    db, rc = built_db
    assert rc == 0
    with sqlite3.connect(db) as con:
        assert fe2c.run_quality_checks(con)["failed"] == 0
    assert len(stints(db)) == 640


def test_simulated_races_are_internally_consistent(built_db):
    s = stints(built_db[0])
    for _, race in s.groupby(["season", "race_round"]):
        fin = race[race["classified"] == 1]
        dnf = race[race["classified"] == 0]
        assert fin["total_laps"].nunique() == 1                       # one race distance
        assert (dnf["total_laps"] < fin["total_laps"].iloc[0]).all()  # DNFs stop early
        assert sorted(race["finish_position"]) == list(range(1, len(race) + 1))
        if len(dnf):
            assert dnf["finish_position"].min() > fin["finish_position"].max()
        f = fin.sort_values("finish_position")
        total_time = (f["total_laps"] * f["avg_lap_sec"]).to_numpy()
        assert (np.diff(total_time) >= 0).all()                      # finishers ordered by race time
    # pole isn't hard-wired to one team any more
    assert s.loc[s["grid_position"] == 1, "team_name"].nunique() > 3
    assert (s["data_source"] == "simulated_seeded").all()


def _metric_frame(planted_team=None):
    """Stints with a known efficiency effect, run through compute_efficiency_metrics."""
    rng = np.random.default_rng(1)
    rows = []
    for race in range(30):
        for team in range(8):
            lap_var = abs(rng.normal(0, 0.8)) + 0.2
            if team == planted_team:
                lap_var *= 0.4  # smoother driving -> higher eer_proxy
            avg_lap, dist = 65 + rng.normal(0, 1.5), 100.0
            rows.append((team + 1, race, lap_var, dist, avg_lap, dist / (avg_lap * lap_var)))
    con = sqlite3.connect(":memory:")
    con.executescript(fe2c.SCHEMA_SQL)
    con.execute("INSERT INTO dim_battery_gen VALUES (3,'Gen3',9,NULL,51,350,1,NULL)")
    con.execute("INSERT INTO dim_track (track_id, circuit_name, city, country, circuit_length_km, "
                "elevation_delta_m) VALUES (1,'T','C','X',2.5,10)")
    for i in range(1, 9):
        con.execute("INSERT INTO dim_manufacturer (manufacturer_id, manufacturer_name) VALUES (?, ?)", (i, f"M{i}"))
    for mid, race, lap_var, dist, avg_lap, eer in rows:
        con.execute(
            "INSERT INTO race_stints (manufacturer_id, track_id, gen_id, season, race_round, race_name, race_date,"
            " driver_name, team_name, lap_time_variance, race_distance_km, avg_lap_sec, avg_lap_velocity_kmh,"
            " eer_proxy, classified) VALUES (?,1,3,9,?, 'R', '2023-01-01', 'D', 'T', ?, ?, ?, ?, ?, 1)",
            (mid, race, lap_var, dist, avg_lap, dist / (avg_lap * 3600) * 3600 * 3.6, eer))
    fe2c.compute_efficiency_metrics(con)
    return pd.read_sql("SELECT manufacturer_id, fe2c_efficiency_score, delta_e FROM race_stints", con)


def test_more_efficient_team_scores_higher_and_lower_delta_e():
    d = _metric_frame(planted_team=0).groupby("manufacturer_id").mean()
    assert d["fe2c_efficiency_score"].idxmax() == 1   # the planted team ranks first, not last
    assert d["delta_e"].idxmin() == 1                 # and uses less energy than the race average


def test_range_ci_includes_baseline_when_there_is_no_effect():
    rng = np.random.default_rng(0)
    rt = fe2c.range_transfer(rng.normal(0.1, 0.03, 40), rng.normal(0.1, 0.03, 280))
    assert rt["p_value"] > 0.05
    assert rt["ci_lo"] < fe2c.LUCID_AIR_EPA_RANGE < rt["ci_hi"]


def test_range_ci_allows_a_loss_and_detects_a_real_gain():
    rng = np.random.default_rng(0)
    worse = fe2c.range_transfer(rng.normal(0.07, 0.02, 40), rng.normal(0.1, 0.02, 280))
    assert worse["simulated_range_mi"] < fe2c.LUCID_AIR_EPA_RANGE and worse["ci_hi"] < fe2c.LUCID_AIR_EPA_RANGE
    better = fe2c.range_transfer(rng.normal(0.13, 0.02, 40), rng.normal(0.1, 0.02, 280))
    assert better["ci_lo"] > fe2c.LUCID_AIR_EPA_RANGE


def test_corrected_reference_figures():
    assert fe2c.LUCID_BENCHMARKS["Air Sapphire"]["epa_range_mi"] == 427
    assert fe2c.ENGINEERED_PROXIES["gen3_recovery_ratio"] == pytest.approx(0.40)
    assert fe2c.ENGINEERED_PROXIES["field_energy_kwh_per_km"] < 1.0


@pytest.mark.parametrize("tool, args", [
    ("tool_query_database", {"sql": "SELECT COUNT(*) AS n FROM race_stints"}),
    ("tool_ca_market_data", {}),
    ("tool_efficiency_comparison", {"team_name": "Mahindra"}),
    ("tool_range_simulation", {"gen_filter": "Gen3"}),
    ("tool_new_circuit_prediction", {"surface_type": "street", "traction_demand": "high"}),
    ("tool_calculate_regen_index", {}),
    ("tool_strategy_recommendation", {"county": "San Mateo"}),
])
def test_agent_tools_run_on_the_built_db(built_db, tool, args, monkeypatch):
    monkeypatch.setattr(fe2c, "DB_PATH", built_db[0])
    out = getattr(fe2c, tool)(args)
    assert isinstance(out, str) and out
    assert "ERROR" not in out.upper().split("\n")[0]
