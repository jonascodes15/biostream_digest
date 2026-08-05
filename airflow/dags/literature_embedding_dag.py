"""Airflow DAG — BioStreamer Track 2 (unstructured literature knowledge).

    chunk_corpus -> embed_and_index -> verify_retrieval

The literature corpus is static reference text (src/common/literature.py), so
this DAG runs far less often than the daily telemetry DAG -- on demand, or
whenever the corpus changes. Re-running it is a no-op: point IDs are
deterministic (uuid5 over source + chunk index), so re-embedding unchanged
text upserts the same points rather than duplicating them.

The pipeline modules underneath are runnable standalone
(``python -m src.pipelines.embed_literature``), so Airflow schedules the work
but does not own it.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

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

# A handful of retrieval smoke tests: (query, expected provenance of the top hit).
# These catch a corpus edit or embedding-model swap that silently degrades
# retrieval quality, without requiring a labelled evaluation set.
_RETRIEVAL_PROBES: list[tuple[str, str]] = [
    ("why did BP1u co-digestion outperform BP2u and single substrates", "published_finding"),
    ("what causes a digester to sour and stop producing gas", "domain_context"),
    ("what was the biogas yield of plantain peel chaff alone", "published_finding"),
]


def task_chunk_corpus(**context) -> int:
    """Chunk the corpus and stage it for embedding. Returns the chunk count."""
    from src.common.literature import CORPUS
    from src.pipelines.embed_literature import chunk_corpus

    chunks = chunk_corpus(CORPUS)
    LOG.info("Chunked %d source documents into %d chunks", len(CORPUS), len(chunks))
    return len(chunks)


def task_embed_and_index(**context) -> None:
    """Embed the corpus and upsert into Qdrant. Idempotent on deterministic IDs."""
    from src.pipelines.embed_literature import run_pipeline

    summary = run_pipeline()
    LOG.info("Index summary: %s", summary)

    if summary["chunks_upserted"] == 0:
        raise ValueError("No chunks were embedded -- corpus may be empty.")
    if summary["published_finding_chunks"] == 0:
        raise ValueError(
            "No published-finding chunks in the index -- the source paper's "
            "text appears to be missing from the corpus."
        )


def task_verify_retrieval(**context) -> None:
    """Run retrieval smoke tests; fail the DAG if provenance ranking regresses.

    This is Track 2's counterpart to Track 1's validate_against_paper gate: a
    cheap, deterministic check that the index still answers the two question
    types the RAG layer depends on -- "what did the paper find" versus
    "what does the literature say about the mechanism" -- correctly.
    """
    from src.pipelines.embed_literature import search

    failures = []
    for query, expected_provenance in _RETRIEVAL_PROBES:
        hits = search(query, top_k=1)
        if not hits:
            failures.append(f"'{query}' -> no hits")
            continue
        top = hits[0]
        if top["provenance"] != expected_provenance:
            failures.append(
                f"'{query}' -> top hit provenance={top['provenance']!r}, "
                f"expected {expected_provenance!r} (score={top['score']})"
            )
        else:
            LOG.info(
                "OK: '%s' -> %s (score=%.3f)", query, top["provenance"], top["score"]
            )

    if failures:
        raise ValueError("Retrieval smoke tests failed:\n" + "\n".join(failures))
    LOG.info("All %d retrieval probes passed.", len(_RETRIEVAL_PROBES))


with DAG(
    dag_id="literature_embedding",
    description=(
        "Chunk, embed, and index the bioprocess literature corpus into Qdrant "
        "(source: DOI 10.30574/gscbps.2024.29.2.0423 + domain process notes)"
    ),
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 10, 1),
    schedule_interval=None,        # on-demand: the corpus is static reference text
    catchup=False,
    max_active_runs=1,
    tags=["biostreamer", "track2", "unstructured", "vector"],
) as dag:

    chunk = PythonOperator(task_id="chunk_corpus", python_callable=task_chunk_corpus)
    embed = PythonOperator(task_id="embed_and_index", python_callable=task_embed_and_index)
    verify = PythonOperator(task_id="verify_retrieval", python_callable=task_verify_retrieval)

    chunk >> embed >> verify
