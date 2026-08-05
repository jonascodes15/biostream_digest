# BioStreamer System Architecture

**A multi-modal data platform for parallel anaerobic digestion research.**

BioStreamer addresses a concrete operational bottleneck from the study *Effects of
slurry concentration and co-digestion on biogas yields from unseeded Phaseolus
vulgaris (bean) peels chaff and unseeded Musa paradisiaca (plantain) peels chaff*
(Nnokwe, Orji, Ajuruchi & Jonas, **GSC Biological and Pharmaceutical Sciences**,
2024, 29(02), 214-218, DOI [10.30574/gscbps.2024.29.2.0423](https://doi.org/10.30574/gscbps.2024.29.2.0423)).
That study ran 36 mini-digesters over 37 days of manual daily water-displacement
readings, transcribed by hand into a spreadsheet. That workflow does not survive
being scaled to 100+ parallel reactor lines. BioStreamer is the data platform that
does.

---

## 1. Scientific ground truth

Every constraint in this platform traces to a measured value in the source study.
These are the anchors the simulator, the schema constraints, and the RAG retrieval
layer are all validated against.

### 1.1 Substrate proximate composition (paper, Table 1)

| Substrate | Code | Ash % | Moisture % | N % | C % | C/N | Total Solids % |
|---|---|---|---|---|---|---|---|
| Bean peel chaff | `Bu` | 13.15 | 3.97 | 3.777 | 34.15 | 9.043 | 96.03 |
| Plantain peel chaff | `Pu` | 15.59 | 9.88 | 2.678 | 37.89 | 14.138 | 90.12 |
| Mixed bean/plantain 1 (0.691 : 1) | `BP1u` | 7.88 | 15.85 | 3.23 | 44.73 | 13.85 | 84.15 |
| Mixed bean/plantain 2 (1 : 1) | `BP2u` | 6.49 | 15.06 | 3.18 | 48.65 | 15.30 | 84.94 |

### 1.2 Mean biogas yield, ml per day (paper, Table 2)

| Substrate | C1 (1:6) | C2 (1:11) | C3 (1:16) | Total |
|---|---|---|---|---|
| `Bu` | 4.40 | 5.06 | 0.34 | 9.80 |
| `Pu` | 0.55 | 0.00 | 0.00 | 0.55 |
| `BP1u` | **8.81** | 1.30 | 1.01 | **11.12** |
| `BP2u` | 3.69 | 1.95 | 1.86 | 7.50 |
| **Total** | **17.45** | 8.31 | 3.21 | 28.97 |

### 1.3 Experimental design

- Randomised complete block design (RCBD), two factors: substrate (4 levels) and
  slurry concentration (3 levels), in triplicate, giving **36 digesters** total.
- 1000 ml mini-digesters, **753 cm³ working slurry volume**, sterilised at 121 degrees C
  for 15 min, sealed, gas collected by water displacement into an inverted calibrated
  cylinder.
- **37-day retention**, daily readings.
- **Unseeded.** No cow rumen liquor or other inoculum. This is the defining
  constraint of the dataset: without a methanogenic seed, digesters exhibit long and
  highly variable lag phases, and some (notably `Pu` at C2 and C3) never produce
  measurable gas at all.
- Two-way ANOVA with replication: substrate F = 11.93 (p = 1.62 x 10 to the -7),
  slurry concentration F = 21.06 (p = 1.87 x 10 to the -9), interaction F = 7.67
  (p = 7.68 x 10 to the -8). All three effects highly significant.

### 1.4 Reconciling slurry ratio with percent total solids

The paper expresses concentration as a substrate-to-water **mass ratio**; the platform
schema stores **percent Total Solids**, the standard AD process variable. The
conversion is:

```
TS_slurry (%) = ( 1 / (1 + r) ) x TS_substrate (%)
```

where `r` is the water parts per part substrate. For bean peel chaff (TS = 96.03%):

| Paper level | Ratio | Substrate mass fraction | Resulting slurry %TS |
|---|---|---|---|
| C1 | 1 : 6 | 0.1429 | **13.72%** |
| C2 | 1 : 11 | 0.0833 | **8.00%** |
| C3 | 1 : 16 | 0.0588 | **5.65%** |

The published design therefore spans roughly 5.6 to 13.7% TS. The platform samples
the wider **5 to 25% TS** design space, and every record carries both columns plus
an `is_reference_design` flag marking the 36 rows that replicate the published RCBD.
Anything above roughly 14% TS is **extrapolation beyond the published envelope** and
is flagged as such wherever it surfaces in the UI and the RAG context.

---

## 2. Platform topology

```
                        +----------------------------------------+
                        |             SOURCE LAYER                |
                        +----------------------------------------+
                                        |
        +---------------------------------+-------------------------------+
        |                                                                 |
  TRACK 1: STRUCTURED                                       TRACK 2: UNSTRUCTURED
  Mock IoT telemetry stream                                Peer-reviewed literature
  (100 reactors x 30 days)                                 (DOI 10.30574/gscbps.2024.29.2.0423)
        |                                                                 |
        v                                                                 v
  +-----------------------+                                 +----------------------------+
  | generate_telemetry    |  mechanistic VFA/pH/Gompertz    |  embed_literature.py       |
  | .py                   |  simulation, seeded RNG          |  langchain-text-splitters  |
  +----------+-------------+                                 |  RecursiveCharacter        |
             |                                               |  chunk=512 overlap=64      |
             | Parquet (bronze)                              +-------------+--------------+
             v                                                             |
  +-----------------------+                                                v
  |  MinIO  :9000         |                                 +----------------------------+
  |  s3://biostreamer-    |                                 | sentence-transformers      |
  |  lake/bronze/         |                                 | all-MiniLM-L6-v2 (384-d)   |
  +----------+-------------+                                 +-------------+--------------+
             | COPY                                                        | upsert
             v                                                             v
  +-----------------------+                                 +----------------------------+
  |  PostgreSQL :5432     |                                 |  Qdrant :6333               |
  |  bioprocess.reactor_  |                                 |  collection:                |
  |    config             |                                 |  bioprocess_knowledge       |
  |  bioprocess.reactor_  |                                 |  COSINE, 384-d              |
  |    telemetry          |                                 +-------------+--------------+
  +----------+-------------+                                              |
             |                                                            |
             +----------------------+-------------------------------------+
                                    v
                        +----------------------------+
                        |  UNIFIED RAG LAYER          |
                        |  FastAPI  :8000              |
                        |  /chat  hybrid retrieve      |
                        |  /reactors                   |
                        |  /telemetry/{id}              |
                        |  Anthropic Messages API       |
                        +-------------+--------------+
                                    v
                        +----------------------------+
                        |  Streamlit  :8501             |
                        |  yield curves + chatbot        |
                        +----------------------------+

  ORCHESTRATION: Apache Airflow :8080, LocalExecutor, DAGs in ./airflow/dags
```

> **Orchestrator selection.** Mage.ai was evaluated as the orchestration layer and
> set aside in favour of **Apache Airflow 2.8.1**. Two reasons decided it. First,
> the scientific gate in this platform (refusing to publish a dataset that has
> stopped reproducing the source study's Table 2) is naturally expressed as a task
> that fails its downstream dependencies, which is Airflow's native model. Second,
> Airflow shares the PostgreSQL instance already provisioned for the warehouse as
> its metadata store, so the stack carries one database rather than two. Pipelines
> are authored as DAGs in `./airflow/dags`, bind-mounted into the container. Both
> pipeline modules also run standalone (`python -m src.pipelines.generate_telemetry`),
> so the orchestrator remains a scheduling concern rather than a hard dependency; the
> science does not depend on which scheduler invokes it.

---

## 3. Track 1: Structured telemetry

### 3.1 Simulation model

`src/pipelines/generate_telemetry.py` does not sprinkle noise on a curve. It runs a
coupled acidogenesis/methanogenesis state model per reactor per day, then calibrates
the magnitude so that the 36 reference reactors reproduce the paper's Table 2 means.

**State variables (per reactor, per day):**

| Variable | Model |
|---|---|
| VFA (mg per L as acetic acid) | Accumulates from hydrolysis at a rate proportional to available TS; consumed by the active methanogen population. |
| Alkalinity (mg per L CaCO3) | Derived from substrate N content (ammonia buffering); higher C/N means lower buffer capacity. |
| pH | Computed from the VFA to alkalinity ratio, not sampled independently. Falls as VFA outpaces buffering; recovers as methanogens establish. |
| Methanogen activity | Logistic growth, **gated by pH**: suppressed below 6.3, effectively zero below 6.0. This is the causal link that makes soured reactors produce no gas. |
| Daily biogas (ml) | Methanogen activity multiplied by substrate-specific potential, shaped by a modified Gompertz envelope. |

**Modified Gompertz** (standard AD kinetics; cf. Latinwo & Agarry, and Olugbemide
*et al.*, both cited in the source paper):

```
M(t) = P * exp{ -exp[ (Rm * e / P)(lambda - t) + 1 ] }
```

`P` is ultimate cumulative yield, `Rm` is peak daily rate, `lambda` is lag phase.
Because the digesters are **unseeded**, lag phase is drawn long and wide (roughly
8 to 22 days) rather than the 2 to 4 days typical of seeded systems. This is what
reproduces `Pu`'s near-zero yields.

**Process state machine:** each reactor-day is labelled with a process state that
progresses from `LAG` to `ACIDOGENIC` to `METHANOGENIC`, or diverts to `SOURED` and
then `RECOVERING` when pH collapses. This label is what the RAG layer reasons over
when asked "which reactors are failing and why."

**Determinism.** A single `numpy.random.default_rng(seed)` drives the whole run.
Re-running with the same seed reproduces the dataset byte for byte, a requirement
for any dataset a paper is going to cite.

**Reactor allocation (100 lines):**
- **36 reference reactors.** Exact replication of the published RCBD
  (4 substrates x 3 concentrations x 3 replicates), `is_reference_design = true`.
- **64 exploratory reactors.** Continuous bean mass fraction between 0 and 1, and
  slurry concentration between 5% and 25% TS, sampling the design space the original
  study could not afford to.

Substrate properties for exploratory blends are **piecewise-linearly interpolated
across the four measured anchor points** (bean fraction 0.0, 0.4086, 0.5, 1.0) rather
than assumed linear mixing, so the interpolant passes exactly through the paper's
measured values, including the non-monotonic carbon content, which a naive linear
mixing model would smooth away.

### 3.2 Warehouse schema

`bioprocess.reactor_config`: one row per reactor (dimension table).

| Column | Type | Notes |
|---|---|---|
| `reactor_id` | `TEXT PK` | `R001` through `R100` |
| `substrate_code` | `TEXT` | `Bu`, `Pu`, `BP1u`, `BP2u`, or `BLEND` |
| `substrate_ratio` | `TEXT` | e.g. `0.691:1` |
| `bean_fraction` | `NUMERIC(5,4)` | 0 to 1 |
| `slurry_ratio` | `TEXT` | e.g. `1:6` |
| `slurry_concentration_pct_ts` | `NUMERIC(5,2)` | **CHECK between 5 and 25** |
| `carbon_nitrogen_ratio` | `NUMERIC(6,3)` | |
| `substrate_total_solids_pct` | `NUMERIC(5,2)` | |
| `working_volume_ml` | `NUMERIC(7,2)` | 753.0 |
| `is_reference_design` | `BOOLEAN` | true for the 36 RCBD replicates |
| `replicate_id` | `SMALLINT` | 1 to 3 |

`bioprocess.reactor_telemetry`: one row per reactor-day (fact table, 100 x 30 = 3,000
rows).

| Column | Type | Notes |
|---|---|---|
| `reading_id` | `BIGSERIAL PK` | |
| `reactor_id` | `TEXT FK` | references `reactor_config` |
| `reading_date` | `DATE` | |
| `day_index` | `SMALLINT` | 1 to 30, **CHECK between 1 and 30** |
| `daily_ph` | `NUMERIC(4,2)` | **CHECK between 3.5 and 9.0** |
| `vfa_mg_l` | `NUMERIC(9,2)` | |
| `alkalinity_mg_caco3_l` | `NUMERIC(9,2)` | |
| `temperature_c` | `NUMERIC(4,1)` | mesophilic ambient, roughly 28 to 33 degrees C |
| `daily_biogas_ml` | `NUMERIC(9,3)` | **CHECK at least 0** |
| `cumulative_biogas_yield_ml` | `NUMERIC(11,3)` | monotone non-decreasing |
| `specific_yield_ml_g_vs` | `NUMERIC(9,4)` | normalised by volatile solids |
| `process_state` | `TEXT` | enum-constrained |
| `ingested_at` | `TIMESTAMPTZ` | |

A unique constraint on `(reactor_id, day_index)` makes the load **idempotent**.
Reruns use `ON CONFLICT DO UPDATE`, so a failed DAG run can simply be re-triggered.

### 3.3 Lake tier

Bronze Parquet lands in MinIO at `s3://biostreamer-lake/bronze/telemetry/run_date=<iso>/`
before the Postgres load. MinIO is optional: if the endpoint is unreachable the
pipeline logs a warning and proceeds to the warehouse, so a developer without the
object store running is not blocked.

---

## 4. Track 2: Unstructured knowledge

`src/pipelines/embed_literature.py` runs in four stages:

1. **Corpus.** The source paper's full text (methods, results, discussion,
   conclusions), the two result tables serialised as prose, the ANOVA table, plus
   curated domain notes on VFA inhibition and TS-regime effects. Each note is tagged
   with provenance so the API can distinguish a *published finding* from
   *literature-general context*.
2. **Chunking.** `langchain_text_splitters.RecursiveCharacterTextSplitter`,
   `chunk_size=512`, `chunk_overlap=64`, separators tuned to keep numeric tables
   intact rather than splitting a row from its header.
3. **Embedding.** `sentence-transformers` `all-MiniLM-L6-v2`: 384 dimensions,
   normalised, roughly 80 MB, CPU-inferable in this Codespace. Batched at 32.
4. **Sink.** Qdrant collection **`bioprocess_knowledge`**, `Distance.COSINE`,
   `size=384`. Payload carries `text`, `source`, `section`, `doi`, `chunk_index`,
   and `provenance`. Deterministic point IDs (UUID5 over source and chunk index)
   make re-embedding idempotent.

---

## 5. Unified RAG layer

`src/api/main.py` is where the two tracks converge. A `/chat` request runs **both**
retrievals and merges them into a single grounded context:

```
question
   |--> Qdrant  bioprocess_knowledge   --> top-k passages + cosine scores
   `--> PostgreSQL  parameterised SQL  --> deterministic aggregates
                                            (per-substrate mean yield,
                                             soured-reactor counts,
                                             pH excursions, yield ranking)
                          |
                          v
              assembled context block
              (literature and live numbers,
               each labelled with its origin)
                          |
                          v
              Anthropic Messages API
              system prompt: answer ONLY from
              supplied context; cite reactor IDs
              and DOI; say "not in the data" rather
              than extrapolate
                          |
                          v
              answer plus citations plus retrieval trace
```

**Why hybrid, not vector-only.** Asking a vector store "what is the mean yield of
BP1u at C1" retrieves a *passage that discusses* yields; it cannot compute over
3,000 telemetry rows. Asking SQL "why did reactor R047 sour" returns numbers with
no mechanism. Numeric questions are answered by SQL, exactly, from the warehouse,
and mechanistic questions by retrieved literature. The LLM composes; it does not
do arithmetic. This is the single largest lever on hallucination rate in this
domain, because a fabricated biogas figure is indistinguishable from a real one to
a reader.

**LLM configuration.** Anthropic Messages API, `ANTHROPIC_API_KEY` read from `.env`,
model selectable via `ANTHROPIC_MODEL`. If no key is present the service **degrades
rather than fails**: `/chat` still returns ranked passages and the SQL aggregate
table, omitting only the synthesis step. Every other endpoint is unaffected.

**Endpoints:** `/health`, `/reactors`, `/reactors/{id}`, `/telemetry/{id}`,
`/stats/summary`, `/search` (vector only), and `/chat` (hybrid plus LLM).

`src/ui/app.py` (Streamlit, port 8501) consumes the API: cumulative yield curves per
reactor with the reference-design cohort highlighted, a pH-versus-VFA overlay, a
substrate by concentration yield heatmap reproducing Table 2, and a chat panel with
its retrieval trace exposed so a reviewer can audit which passage and which query
produced each claim.

---

## 6. Service inventory

| Service | Port | Image | Role |
|---|---|---|---|
| `digest_postgres` | 5432 | `postgres:15-alpine` | Staging warehouse and Airflow metadata |
| `digest_qdrant` | 6333 | `qdrant/qdrant:latest` | Vector store |
| `digest_minio` | 9000 / 9001 | `minio/minio:latest` | S3-compatible lake |
| `digest_airflow` | 8080 | `apache/airflow:2.8.1-python3.10` | Orchestration (LocalExecutor) |
| FastAPI | 8000 | local `uvicorn` | Unified RAG API |
| Streamlit | 8501 | local | Analyst UI |

Credentials for the local stack are in `docker-compose.yml`. Application secrets go
in `.env` (git-ignored, template in `.env.example`).

---

## 7. Data flow guarantees

| Property | Mechanism |
|---|---|
| **Reproducible** | A single seeded RNG means the same seed always produces an identical dataset. |
| **Idempotent** | `ON CONFLICT (reactor_id, day_index) DO UPDATE`, and UUID5 Qdrant point IDs. |
| **Traceable** | Every telemetry row carries `is_reference_design`; every vector carries `doi` and `section`. |
| **Degradable** | MinIO is optional and LLM is optional; each failure narrows functionality without stopping the pipeline. |
| **Validated** | Reference-cohort yields are asserted against the paper's Table 2 at load time. |

---

## 8. Repository layout

```
biostreamer/
+-- ARCHITECTURE.md              (this document)
+-- TECHNICAL_PAPER.md           (publication draft)
+-- docker-compose.yml
+-- requirements.txt
+-- .env.example
+-- airflow/dags/
|   +-- bioreactor_telemetry_dag.py
|   `-- literature_embedding_dag.py
`-- src/
    +-- common/
    |   +-- config.py            (env-driven settings)
    |   `-- science.py           (paper constants, single source of truth)
    +-- db/schema.sql
    +-- pipelines/
    |   +-- generate_telemetry.py
    |   +-- load_telemetry.py
    |   `-- embed_literature.py
    +-- api/main.py
    `-- ui/app.py
```

`src/common/science.py` holds the paper's measured constants in one place. The
simulator, the schema constraints, the validation assertions, and the RAG corpus
all import from it, so a transcription error can only ever be made once.
