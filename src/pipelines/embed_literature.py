"""Track 2: unstructured knowledge pipeline: chunk, embed, and index the
literature corpus into Qdrant.

    corpus (paper text + domain notes, tagged by provenance)
        -> RecursiveCharacterTextSplitter  (chunk_size=512, overlap=64)
        -> sentence-transformers all-MiniLM-L6-v2  (384-d, normalised, batch 32)
        -> Qdrant collection "bioprocess_knowledge"  (COSINE, size=384)

Determinism: point IDs are uuid5(source_hash + chunk_index), so re-running the
pipeline on unchanged source text is a no-op upsert rather than a duplicate
insert -- the same idempotency contract as the Track 1 warehouse load.

Usage
-----
    python -m src.pipelines.embed_literature              # embed + upsert
    python -m src.pipelines.embed_literature --search "why did BP1u outperform BP2u"
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.common.config import get_settings
from src.common.literature import CORPUS, CorpusDocument

LOG = logging.getLogger("biostreamer.embed")

# Namespace UUID for deterministic point-ID generation. Fixed and arbitrary --
# only its stability across runs matters.
_POINT_NAMESPACE = uuid.UUID("6f6b1a6e-6c4e-4a2b-9b8a-1d2e3f4a5b6c")

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #

def chunk_corpus(corpus: list[CorpusDocument]) -> list[dict]:
    """Split the corpus into embeddable chunks, carrying provenance forward.

    Separators are ordered so a split prefers a paragraph or sentence boundary
    over an arbitrary character cut, which keeps a table row from being torn
    away from the header/label context it needs to be meaningful in isolation.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        length_function=len,
    )

    chunks: list[dict] = []
    for doc_index, doc in enumerate(corpus):
        pieces = splitter.split_text(doc.text)
        for chunk_index, piece in enumerate(pieces):
            chunks.append(
                {
                    "text": piece.strip(),
                    "section": doc.section,
                    "provenance": doc.provenance,
                    "source": doc.source,
                    "doi": doc.doi,
                    "doc_index": doc_index,
                    "chunk_index": chunk_index,
                }
            )

    LOG.info("Chunked %d source documents into %d chunks", len(corpus), len(chunks))
    return chunks


def _point_id(chunk: dict) -> str:
    """Deterministic point ID: same source text + chunk position -> same ID."""
    key = f"{chunk['source']}::{chunk['doc_index']}::{chunk['chunk_index']}"
    return str(uuid.uuid5(_POINT_NAMESPACE, key))


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #

def embed_chunks(chunks: list[dict], model_name: str, batch_size: int = 32) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer

    LOG.info("Loading embedding model %s", model_name)
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,       # required for COSINE distance in Qdrant
        show_progress_bar=False,
    )
    LOG.info("Embedded %d chunks (dim=%d)", len(embeddings), embeddings.shape[1])
    return embeddings.tolist()


# --------------------------------------------------------------------------- #
# Qdrant sink
# --------------------------------------------------------------------------- #

def ensure_collection(client, name: str, dim: int) -> None:
    from qdrant_client.http import models as qm

    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        LOG.info("Collection %s already exists", name)
        return
    client.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )
    LOG.info("Created collection %s (dim=%d, distance=COSINE)", name, dim)


def upsert_chunks(client, collection: str, chunks: list[dict], vectors: list[list[float]]) -> int:
    from qdrant_client.http import models as qm

    points = [
        qm.PointStruct(
            id=_point_id(chunk),
            vector=vector,
            payload={
                "text": chunk["text"],
                "section": chunk["section"],
                "provenance": chunk["provenance"],
                "source": chunk["source"],
                "doi": chunk["doi"],
                "chunk_index": chunk["chunk_index"],
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    client.upsert(collection_name=collection, points=points, wait=True)
    LOG.info("Upserted %d points into %s", len(points), collection)
    return len(points)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_pipeline() -> dict:
    from qdrant_client import QdrantClient

    settings = get_settings()

    chunks = chunk_corpus(CORPUS)
    vectors = embed_chunks(chunks, settings.embedding_model)

    client = QdrantClient(url=settings.qdrant.url)
    ensure_collection(client, settings.qdrant.collection, settings.embedding_dim)
    n_upserted = upsert_chunks(client, settings.qdrant.collection, chunks, vectors)

    info = client.get_collection(settings.qdrant.collection)
    summary = {
        "chunks_upserted": n_upserted,
        "collection_points": info.points_count,
        "published_finding_chunks": sum(1 for c in chunks if c["provenance"] == "published_finding"),
        "domain_context_chunks": sum(1 for c in chunks if c["provenance"] == "domain_context"),
    }
    return summary


def search(query: str, top_k: int = 5) -> list[dict]:
    """Ad-hoc semantic search against the live collection, for smoke-testing."""
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    settings = get_settings()
    model = SentenceTransformer(settings.embedding_model)
    vector = model.encode(query, normalize_embeddings=True).tolist()

    client = QdrantClient(url=settings.qdrant.url)
    hits = client.query_points(
        collection_name=settings.qdrant.collection, query=vector, limit=top_k
    ).points

    return [
        {
            "score": round(h.score, 4),
            "section": h.payload["section"],
            "provenance": h.payload["provenance"],
            "text": h.payload["text"],
        }
        for h in hits
    ]


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    ap = argparse.ArgumentParser(description="Embed the BioStreamer literature corpus into Qdrant.")
    ap.add_argument("--search", type=str, default=None, help="Skip indexing; run a test query instead")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    if args.search:
        for i, hit in enumerate(search(args.search, args.top_k), start=1):
            print(f"\n[{i}] score={hit['score']}  section={hit['section']}  provenance={hit['provenance']}")
            print(f"    {hit['text'][:220]}{'...' if len(hit['text']) > 220 else ''}")
        return 0

    summary = run_pipeline()
    print("\n--- BioStreamer literature index ---")
    for key, value in summary.items():
        print(f"  {key:28s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
