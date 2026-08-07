"""BioStreamer unified API: the point where both tracks converge.

    /reactors, /telemetry/{id}, /stats/summary   -> structured warehouse (SQL)
    /search                                       -> vector store (Qdrant) only
    /chat                                         -> hybrid: both retrieved,
                                                      merged into one grounded
                                                      context, handed to Claude

The hybrid design is deliberate. A vector store retrieves a passage that
*discusses* yields; it cannot compute a mean over 3,700 telemetry rows. SQL
returns exact numbers with no mechanism. Numeric questions are answered from
the warehouse, mechanistic questions from retrieved literature, and the LLM
composes the two, and never does the arithmetic itself. That split is the
single largest lever on hallucination risk in this domain, because a
fabricated yield figure looks identical to a real one.

Degrades gracefully: if ANTHROPIC_API_KEY is unset, /chat still returns the
retrieved passages and SQL aggregates, omitting only the synthesis step.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.common import science as sci
from src.common.config import Settings, get_settings

LOG = logging.getLogger("biostreamer.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

_state: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# Lifecycle: lazy-init clients once, reuse across requests
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _state["settings"] = settings

    try:
        from qdrant_client import QdrantClient
        _state["qdrant"] = QdrantClient(url=settings.qdrant.url)
        LOG.info("Qdrant client ready (%s)", settings.qdrant.url)
    except Exception as exc:                       # noqa: BLE001
        LOG.warning("Qdrant unavailable at startup: %s", exc)
        _state["qdrant"] = None

    try:
        from sentence_transformers import SentenceTransformer
        _state["embedder"] = SentenceTransformer(settings.embedding_model)
        LOG.info("Embedding model loaded (%s)", settings.embedding_model)
    except Exception as exc:                        # noqa: BLE001
        LOG.warning("Embedding model unavailable at startup: %s", exc)
        _state["embedder"] = None

    if settings.llm.enabled:
        try:
            import anthropic
            _state["llm"] = anthropic.Anthropic(api_key=settings.llm.api_key)
            LOG.info("Anthropic client ready (model=%s)", settings.llm.model)
        except Exception as exc:                    # noqa: BLE001
            LOG.warning("Anthropic client unavailable: %s", exc)
            _state["llm"] = None
    else:
        LOG.warning("ANTHROPIC_API_KEY unset -- /chat will degrade to retrieval-only")
        _state["llm"] = None

    yield
    _state.clear()


app = FastAPI(
    title="BioStreamer API",
    description=(
        "Unified access to structured bioreactor telemetry and unstructured "
        "bioprocess literature, grounded in DOI 10.30574/gscbps.2024.29.2.0423."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# DB helper
# --------------------------------------------------------------------------- #

def _db(settings: Settings):
    conn = psycopg2.connect(settings.postgres.dsn)
    conn.cursor_factory = psycopg2.extras.RealDictCursor
    return conn


def _query(sql: str, params: tuple = ()) -> list[dict]:
    settings: Settings = _state["settings"]
    with _db(settings) as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


# --------------------------------------------------------------------------- #
# Response models
# --------------------------------------------------------------------------- #

class ReactorSummary(BaseModel):
    reactor_id: str
    substrate_code: str
    substrate_ratio: str
    slurry_ratio: str
    slurry_concentration_pct_ts: float
    carbon_nitrogen_ratio: float
    is_reference_design: bool
    beyond_published_envelope: bool
    final_cumulative_ml: float | None = None
    mean_ph: float | None = None
    soured_days: int | None = None


class TelemetryReading(BaseModel):
    day_index: int
    reading_date: str
    daily_ph: float
    daily_biogas_ml: float
    cumulative_biogas_yield_ml: float
    process_state: str


class SearchHit(BaseModel):
    score: float
    section: str
    provenance: Literal["published_finding", "domain_context"]
    text: str
    doi: str | None = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str | None
    llm_used: bool
    literature_context: list[SearchHit]
    sql_context: dict[str, Any]
    note: str | None = None


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #

@app.get("/health")
def health() -> dict:
    settings: Settings = _state["settings"]
    checks: dict[str, bool] = {}

    try:
        _query("SELECT 1")
        checks["postgres"] = True
    except Exception:                               # noqa: BLE001
        checks["postgres"] = False

    checks["qdrant"] = _state.get("qdrant") is not None
    checks["embedder"] = _state.get("embedder") is not None
    checks["llm"] = _state.get("llm") is not None

    return {
        "status": "ok" if all(checks.values()) else "degraded",
        "checks": checks,
        "source_paper_doi": sci.PAPER_DOI,
    }


# --------------------------------------------------------------------------- #
# Structured (SQL) endpoints
# --------------------------------------------------------------------------- #

@app.get("/reactors", response_model=list[ReactorSummary])
def list_reactors(
    substrate: str | None = None,
    reference_only: bool = False,
    limit: int = Query(default=100, le=100),
) -> list[dict]:
    clauses, params = [], []
    if substrate:
        clauses.append("substrate_code = %s")
        params.append(substrate)
    if reference_only:
        clauses.append("is_reference_design = TRUE")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    rows = _query(
        f"""
        SELECT reactor_id, substrate_code, substrate_ratio, slurry_ratio,
               slurry_concentration_pct_ts, carbon_nitrogen_ratio,
               is_reference_design, beyond_published_envelope,
               final_cumulative_ml, mean_ph, soured_days
        FROM bioprocess.v_reactor_performance
        {where}
        ORDER BY reactor_id
        LIMIT %s
        """,
        (*params, limit),
    )
    if not rows:
        raise HTTPException(404, "No reactors found for the given filters.")
    return rows


@app.get("/reactors/{reactor_id}")
def get_reactor(reactor_id: str) -> dict:
    rows = _query(
        "SELECT * FROM bioprocess.v_reactor_performance WHERE reactor_id = %s",
        (reactor_id,),
    )
    if not rows:
        raise HTTPException(404, f"Reactor {reactor_id} not found.")
    return rows[0]


@app.get("/telemetry/{reactor_id}", response_model=list[TelemetryReading])
def get_telemetry(reactor_id: str) -> list[dict]:
    rows = _query(
        """
        SELECT day_index, reading_date::text, daily_ph, daily_biogas_ml,
               cumulative_biogas_yield_ml, process_state
        FROM bioprocess.reactor_telemetry
        WHERE reactor_id = %s
        ORDER BY day_index
        """,
        (reactor_id,),
    )
    if not rows:
        raise HTTPException(404, f"No telemetry for reactor {reactor_id}.")
    return rows


@app.get("/stats/summary")
def stats_summary() -> dict:
    reference_matrix = _query(
        "SELECT * FROM bioprocess.v_reference_yield_matrix ORDER BY substrate_code, slurry_level_code"
    )
    totals = _query(
        """
        SELECT count(*) AS reactor_count,
               round(avg(final_cumulative_ml)::numeric, 2) AS avg_final_yield_ml,
               round(min(min_ph)::numeric, 2) AS worst_min_ph,
               sum(soured_days) AS total_soured_reactor_days
        FROM bioprocess.v_reactor_performance
        """
    )[0]
    return {
        "reference_yield_matrix": reference_matrix,
        "published_grand_total_ml_day": sci.GRAND_TOTAL_ML_DAY,
        "warehouse_totals": totals,
        "source_paper_doi": sci.PAPER_DOI,
    }


@app.get("/alerts")
def process_alerts(limit: int = Query(default=50, le=200)) -> list[dict]:
    return _query(
        """
        SELECT reactor_id, substrate_code, slurry_concentration_pct_ts,
               beyond_published_envelope, day_index, reading_date::text,
               daily_ph, vfa_alkalinity_ratio, process_state
        FROM bioprocess.v_process_alerts
        ORDER BY reading_date DESC, reactor_id
        LIMIT %s
        """,
        (limit,),
    )


# --------------------------------------------------------------------------- #
# Vector retrieval
# --------------------------------------------------------------------------- #

def _vector_search(query: str, top_k: int) -> list[SearchHit]:
    embedder = _state.get("embedder")
    qdrant = _state.get("qdrant")
    if embedder is None or qdrant is None:
        return []

    settings: Settings = _state["settings"]
    vector = embedder.encode(query, normalize_embeddings=True).tolist()
    hits = qdrant.query_points(
        collection_name=settings.qdrant.collection, query=vector, limit=top_k
    ).points
    return [
        SearchHit(
            score=round(h.score, 4),
            section=h.payload["section"],
            provenance=h.payload["provenance"],
            text=h.payload["text"],
            doi=h.payload.get("doi"),
        )
        for h in hits
    ]


@app.get("/search", response_model=list[SearchHit])
def search(q: str = Query(..., min_length=1), top_k: int = Query(default=5, le=20)) -> list[SearchHit]:
    hits = _vector_search(q, top_k)
    if not hits:
        raise HTTPException(503, "Vector search unavailable (Qdrant or embedding model not ready).")
    return hits


# --------------------------------------------------------------------------- #
# Hybrid SQL context for /chat
# --------------------------------------------------------------------------- #

def _sql_context_for_question(question: str) -> dict[str, Any]:
    """Deterministic aggregates likely relevant to any bioprocess question.

    Not NL-to-SQL -- a fixed set of warehouse aggregates is assembled every
    time, cheaply, and handed to the LLM alongside the literature context. The
    model composes an answer from real numbers; it never invents one.
    """
    reference_matrix = _query(
        "SELECT * FROM bioprocess.v_reference_yield_matrix ORDER BY substrate_code, slurry_level_code"
    )
    by_substrate = _query(
        """
        SELECT substrate_code,
               count(*) AS reactors,
               round(avg(final_cumulative_ml)::numeric, 1) AS avg_final_yield_ml,
               round(avg(mean_ph)::numeric, 2) AS avg_ph,
               sum(soured_days) AS soured_days
        FROM bioprocess.v_reactor_performance
        GROUP BY substrate_code
        ORDER BY avg_final_yield_ml DESC
        """
    )
    alerts = _query(
        "SELECT count(*) AS alert_count FROM bioprocess.v_process_alerts"
    )[0]

    return {
        "reference_cohort_vs_published_table2": reference_matrix,
        "performance_by_substrate": by_substrate,
        "active_process_alerts": alerts["alert_count"],
    }


# --------------------------------------------------------------------------- #
# /chat -- the unified RAG endpoint
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = f"""You are the BioStreamer research assistant for a bioprocess \
data platform studying anaerobic digestion of bean and plantain peel waste.

You are given two kinds of context for every question:
  1. LITERATURE PASSAGES, each tagged provenance="published_finding" (drawn \
directly from the source paper, DOI {sci.PAPER_DOI}) or \
provenance="domain_context" (general anaerobic-digestion process knowledge, \
NOT measured in that paper).
  2. WAREHOUSE DATA: exact aggregates computed from the live telemetry database.

Rules:
- Answer ONLY from the supplied context. Do not use outside knowledge of \
biogas research beyond what is provided.
- When citing a measured result, say whether it comes from the published \
paper or from the simulated warehouse, and cite reactor IDs or the DOI where \
relevant.
- Never present a domain_context passage as something the source paper measured.
- Any reactor with slurry_concentration_pct_ts above {sci.PUBLISHED_TS_CEILING_PCT}% \
is beyond the published experimental envelope (which spans ~5.6-13.7% TS) -- \
say so explicitly if such a reactor is discussed.
- If the supplied context does not contain the answer, say so plainly rather \
than extrapolating or guessing a number.
- Be concise. Lead with the answer."""


def _build_llm_context(question: str, literature: list[SearchHit], sql: dict) -> str:
    lit_block = "\n\n".join(
        f"[{i+1}] (provenance={h.provenance}, section={h.section}, score={h.score})\n{h.text}"
        for i, h in enumerate(literature)
    ) or "(no literature passages retrieved)"

    return (
        f"Question: {question}\n\n"
        f"--- LITERATURE PASSAGES ---\n{lit_block}\n\n"
        f"--- WAREHOUSE DATA (JSON) ---\n{sql}"
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    literature = _vector_search(req.question, req.top_k)
    sql_context = _sql_context_for_question(req.question)

    llm = _state.get("llm")
    if llm is None:
        return ChatResponse(
            answer=None,
            llm_used=False,
            literature_context=literature,
            sql_context=sql_context,
            note=(
                "ANTHROPIC_API_KEY is not configured, so synthesis is skipped. "
                "Retrieved literature passages and warehouse aggregates are "
                "returned below; set ANTHROPIC_API_KEY in .env to enable answers."
            ),
        )

    settings: Settings = _state["settings"]
    user_content = _build_llm_context(req.question, literature, sql_context)

    try:
        response = llm.messages.create(
            model=settings.llm.model,
            max_tokens=settings.llm.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        answer = next((b.text for b in response.content if b.type == "text"), "")
    except Exception as exc:                        # noqa: BLE001
        LOG.error("LLM call failed: %s", exc)
        return ChatResponse(
            answer=None,
            llm_used=False,
            literature_context=literature,
            sql_context=sql_context,
            note=f"LLM call failed ({exc}); returning retrieval context only.",
        )

    return ChatResponse(
        answer=answer,
        llm_used=True,
        literature_context=literature,
        sql_context=sql_context,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
