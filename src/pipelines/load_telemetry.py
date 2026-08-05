"""Track 1 sink — land telemetry in MinIO (bronze) and PostgreSQL (silver).

Idempotent by construction: the warehouse load is an upsert keyed on
(reactor_id, day_index), so a failed or re-triggered DAG run converges to the
same state rather than duplicating rows.

MinIO is optional. If the object store is unreachable the pipeline logs a
warning and proceeds to the warehouse -- a developer without the lake running
is not blocked.

Usage
-----
    python -m src.pipelines.load_telemetry            # generate + load
    python -m src.pipelines.load_telemetry --verify   # report what landed
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import science as sci
from src.common.config import Settings, get_settings
from src.pipelines.generate_telemetry import generate, validate_against_paper

LOG = logging.getLogger("biostreamer.load")

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "db" / "schema.sql"

CONFIG_COLUMNS = [
    "reactor_id", "substrate_code", "substrate_name", "substrate_ratio",
    "bean_fraction", "plantain_fraction", "slurry_level_code", "slurry_ratio",
    "slurry_concentration_pct_ts", "carbon_nitrogen_ratio",
    "substrate_total_solids_pct", "substrate_nitrogen_pct", "substrate_ash_pct",
    "working_volume_ml", "is_reference_design", "beyond_published_envelope",
    "replicate_id",
]

TELEMETRY_COLUMNS = [
    "reactor_id", "reading_date", "day_index", "daily_ph", "vfa_mg_l",
    "alkalinity_mg_caco3_l", "temperature_c", "daily_biogas_ml",
    "cumulative_biogas_yield_ml", "specific_yield_ml_g_vs",
    "methanogen_activity", "process_state", "ingested_at",
]


# --------------------------------------------------------------------------- #
# Bronze tier: MinIO
# --------------------------------------------------------------------------- #

def write_to_lake(
    configs: pd.DataFrame,
    telemetry: pd.DataFrame,
    settings: Settings,
    run_date: date | None = None,
) -> bool:
    """Write bronze Parquet to MinIO. Returns False (non-fatal) if unavailable."""
    try:
        from minio import Minio
    except ImportError:
        LOG.warning("minio package not installed - skipping lake tier")
        return False

    run_date = run_date or date.today()
    cfg = settings.minio

    try:
        client = Minio(
            cfg.endpoint,
            access_key=cfg.access_key,
            secret_key=cfg.secret_key,
            secure=cfg.secure,
        )
        if not client.bucket_exists(cfg.bucket):
            client.make_bucket(cfg.bucket)
            LOG.info("Created bucket %s", cfg.bucket)

        for name, df in (("reactor_config", configs), ("reactor_telemetry", telemetry)):
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            buf.seek(0)
            key = f"bronze/{name}/run_date={run_date.isoformat()}/{name}.parquet"
            client.put_object(
                cfg.bucket, key, buf, length=buf.getbuffer().nbytes,
                content_type="application/octet-stream",
            )
            LOG.info("Wrote s3://%s/%s (%d rows)", cfg.bucket, key, len(df))
        return True
    except Exception as exc:                      # noqa: BLE001 - lake is optional
        LOG.warning("Lake tier unavailable (%s) - continuing to warehouse", exc)
        return False


# --------------------------------------------------------------------------- #
# Silver tier: PostgreSQL
# --------------------------------------------------------------------------- #

def ensure_schema(conn, settings: Settings) -> None:
    ddl = SCHEMA_PATH.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    LOG.info("Schema ensured (%s)", settings.postgres.schema)


def _upsert(conn, table: str, columns: list[str], df: pd.DataFrame,
            conflict: str, batch: int = 1000) -> int:
    """Upsert a DataFrame, updating every non-key column on conflict."""
    updatable = [c for c in columns if c not in conflict]
    set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in updatable)
    sql = (
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES %s "
        f"ON CONFLICT ({conflict}) DO UPDATE SET {set_clause}"
    )
    # Normalise NaN/NaT to None so psycopg2 emits SQL NULL.
    records = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in df[columns].itertuples(index=False, name=None)
    ]
    with conn.cursor() as cur:
        execute_values(cur, sql, records, page_size=batch)
    conn.commit()
    return len(records)


def load_to_postgres(
    configs: pd.DataFrame, telemetry: pd.DataFrame, settings: Settings
) -> tuple[int, int]:
    with psycopg2.connect(settings.postgres.dsn) as conn:
        ensure_schema(conn, settings)

        n_cfg = _upsert(
            conn, "bioprocess.reactor_config", CONFIG_COLUMNS, configs,
            conflict="reactor_id",
        )
        LOG.info("Upserted %d reactor_config rows", n_cfg)

        n_tel = _upsert(
            conn, "bioprocess.reactor_telemetry", TELEMETRY_COLUMNS, telemetry,
            conflict="reactor_id, day_index",
        )
        LOG.info("Upserted %d reactor_telemetry rows", n_tel)

    return n_cfg, n_tel


def verify(settings: Settings) -> dict:
    """Post-load assertions, including reproduction of the published Table 2."""
    out: dict = {}
    with psycopg2.connect(settings.postgres.dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM bioprocess.reactor_config")
        out["reactors"] = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM bioprocess.reactor_telemetry")
        out["telemetry_rows"] = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(*) FROM bioprocess.reactor_config WHERE is_reference_design"
        )
        out["reference_reactors"] = cur.fetchone()[0]

        cur.execute(
            "SELECT MIN(daily_ph), MAX(daily_ph), "
            "       MIN(daily_biogas_ml), MAX(cumulative_biogas_yield_ml) "
            "FROM bioprocess.reactor_telemetry"
        )
        lo_ph, hi_ph, lo_gas, hi_cum = cur.fetchone()
        out["ph_range"] = (float(lo_ph), float(hi_ph))
        out["min_daily_biogas_ml"] = float(lo_gas)
        out["max_cumulative_ml"] = float(hi_cum)

        cur.execute(
            "SELECT substrate_code, slurry_level_code, observed_mean_ml_day "
            "FROM bioprocess.v_reference_yield_matrix "
            "ORDER BY substrate_code, slurry_level_code"
        )
        matrix = []
        for sub, level, observed in cur.fetchall():
            published = sci.MEAN_DAILY_YIELD_ML[(sub, level)]
            obs = float(observed)
            ok = obs <= 0.35 if published == 0 else abs(obs - published) / published <= 0.25
            matrix.append(
                {
                    "substrate": sub,
                    "level": level,
                    "observed": round(obs, 3),
                    "published": published,
                    "within_tolerance": ok,
                }
            )
        out["reference_matrix"] = matrix
        out["reference_matrix_ok"] = all(m["within_tolerance"] for m in matrix)

        cur.execute("SELECT COUNT(*) FROM bioprocess.v_process_alerts")
        out["alert_rows"] = cur.fetchone()[0]

    return out


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run_pipeline(settings: Settings | None = None, skip_lake: bool = False) -> dict:
    """Full Track 1 pipeline: generate -> validate -> lake -> warehouse -> verify."""
    settings = settings or get_settings()

    configs, telemetry = generate()

    report = validate_against_paper(configs, telemetry)
    failures = report[~report["within_tolerance"]]
    if len(failures):
        raise ValueError(
            "Simulated reference cohort does not reproduce the published "
            f"Table 2 means:\n{failures.to_string(index=False)}"
        )
    LOG.info("Reference cohort reproduces all 12 published cells.")

    if not skip_lake:
        write_to_lake(configs, telemetry, settings)

    load_to_postgres(configs, telemetry, settings)
    return verify(settings)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    ap = argparse.ArgumentParser(description="Load BioStreamer telemetry.")
    ap.add_argument("--skip-lake", action="store_true", help="Skip the MinIO bronze write")
    ap.add_argument("--verify", action="store_true", help="Verify only; do not reload")
    args = ap.parse_args()

    settings = get_settings()
    summary = verify(settings) if args.verify else run_pipeline(settings, args.skip_lake)

    print("\n--- BioStreamer warehouse ---")
    for key in ("reactors", "reference_reactors", "telemetry_rows",
                "ph_range", "min_daily_biogas_ml", "max_cumulative_ml", "alert_rows"):
        print(f"  {key:24s} {summary[key]}")

    print("\n  Reference cohort vs published Table 2 (ml/day):")
    for m in summary["reference_matrix"]:
        flag = "ok " if m["within_tolerance"] else "OUT"
        print(
            f"    [{flag}] {m['substrate']:5s} {m['level']}  "
            f"observed={m['observed']:7.3f}  published={m['published']:6.2f}"
        )

    ok = summary["reference_matrix_ok"]
    print(f"\n  Reproduces published results: {'YES' if ok else 'NO'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
