"""Airflow DAG for BioStreamer Track 1 (structured bioreactor telemetry).

    simulate -> validate_against_paper -> land_bronze -> load_warehouse -> verify

Every task is idempotent, so a failed run can be cleared and re-triggered
without duplicating warehouse rows or corrupting the lake.

The pipeline modules underneath are runnable standalone
(``python -m src.pipelines.load_telemetry``), so Airflow schedules the work but
does not own it -- the science does not depend on which scheduler invokes it.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# The repository is bind-mounted into the Airflow container; make src importable.
PROJECT_ROOT = os.environ.get("BIOSTREAM_PROJECT_ROOT", "/opt/airflow/biostreamer")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LOG = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "bioprocess-data-eng",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #

def task_simulate(**context) -> str:
    """Generate the reactor design and 37-day telemetry; stage as Parquet.

    Parquet is written to a run-scoped staging path rather than passed through
    XCom (3,700 rows of telemetry has no business travelling through the
    metadata database).
    """
    from pathlib import Path

    from src.pipelines.generate_telemetry import generate

    run_id = context["run_id"].replace(":", "-").replace("+", "-")
    staging = Path(PROJECT_ROOT) / "data" / "staging" / run_id
    staging.mkdir(parents=True, exist_ok=True)

    configs, telemetry = generate()
    configs.to_parquet(staging / "reactor_config.parquet", index=False)
    telemetry.to_parquet(staging / "reactor_telemetry.parquet", index=False)

    LOG.info(
        "Staged %d reactors / %d telemetry rows at %s",
        len(configs), len(telemetry), staging,
    )
    return str(staging)


def _read_staged(ti):
    import pandas as pd
    from pathlib import Path

    staging = Path(ti.xcom_pull(task_ids="simulate"))
    return (
        pd.read_parquet(staging / "reactor_config.parquet"),
        pd.read_parquet(staging / "reactor_telemetry.parquet"),
    )


def task_validate(**context) -> None:
    """Fail the run if the reference cohort drifts from the published Table 2.

    This is the DAG's scientific gate. A simulation change that silently stops
    reproducing DOI 10.30574/gscbps.2024.29.2.0423 must not reach the warehouse.
    """
    from src.pipelines.generate_telemetry import validate_against_paper

    configs, telemetry = _read_staged(context["ti"])
    report = validate_against_paper(configs, telemetry)

    LOG.info("Reference cohort vs published Table 2:\n%s", report.to_string(index=False))

    failures = report[~report["within_tolerance"]]
    if len(failures):
        raise ValueError(
            "Reference cohort no longer reproduces the published means:\n"
            f"{failures.to_string(index=False)}"
        )
    LOG.info("All %d reference cells within tolerance.", len(report))


def task_land_bronze(**context) -> None:
    """Write bronze Parquet to the MinIO data lake (non-fatal if unavailable)."""
    from src.common.config import get_settings
    from src.pipelines.load_telemetry import write_to_lake

    configs, telemetry = _read_staged(context["ti"])
    ok = write_to_lake(configs, telemetry, get_settings())
    if not ok:
        LOG.warning("Lake tier skipped; warehouse load will proceed regardless.")


def task_load_warehouse(**context) -> None:
    """Upsert into PostgreSQL. Idempotent on (reactor_id, day_index)."""
    from src.common.config import get_settings
    from src.pipelines.load_telemetry import load_to_postgres

    configs, telemetry = _read_staged(context["ti"])
    n_cfg, n_tel = load_to_postgres(configs, telemetry, get_settings())
    LOG.info("Warehouse load complete: %d configs, %d telemetry rows", n_cfg, n_tel)


def task_verify(**context) -> None:
    """Post-load data-quality assertions against the live warehouse."""
    from src.common.config import get_settings
    from src.pipelines.load_telemetry import verify

    summary = verify(get_settings())
    LOG.info("Warehouse summary: %s", summary)

    if summary["telemetry_rows"] == 0:
        raise ValueError("No telemetry rows present after load.")
    if summary["reference_reactors"] != 36:
        raise ValueError(
            f"Expected 36 reference reactors, found {summary['reference_reactors']}"
        )
    if not summary["reference_matrix_ok"]:
        raise ValueError("Warehouse contents do not reproduce the published Table 2.")

    LOG.info(
        "Verified: %d reactors, %d telemetry rows, %d alert rows.",
        summary["reactors"], summary["telemetry_rows"], summary["alert_rows"],
    )


# --------------------------------------------------------------------------- #
# DAG
# --------------------------------------------------------------------------- #

with DAG(
    dag_id="bioreactor_telemetry",
    description=(
        "Simulate and load daily telemetry for 100 parallel anaerobic digesters "
        "(grounded in DOI 10.30574/gscbps.2024.29.2.0423)"
    ),
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 10, 1),
    schedule_interval="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["biostreamer", "track1", "structured", "bioprocess"],
) as dag:

    simulate = PythonOperator(task_id="simulate", python_callable=task_simulate)
    validate = PythonOperator(task_id="validate_against_paper", python_callable=task_validate)
    bronze = PythonOperator(task_id="land_bronze", python_callable=task_land_bronze)
    warehouse = PythonOperator(task_id="load_warehouse", python_callable=task_load_warehouse)
    verify_load = PythonOperator(task_id="verify", python_callable=task_verify)

    simulate >> validate >> bronze >> warehouse >> verify_load
