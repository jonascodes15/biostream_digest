-- BioStreamer staging warehouse schema.
--
-- Constraints encode the scientific envelope of the source study:
--   Nnokwe JC, Orji MU, Ajuruchi VC, Jonas KC. GSC Biol. Pharm. Sci., 2024,
--   29(02), 214-218. DOI: 10.30574/gscbps.2024.29.2.0423
--
-- Idempotent: safe to run on every DAG execution.

CREATE SCHEMA IF NOT EXISTS bioprocess;

-- --------------------------------------------------------------------------
-- Dimension: one row per reactor line
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bioprocess.reactor_config (
    reactor_id                   TEXT         PRIMARY KEY,
    substrate_code               TEXT         NOT NULL,
    substrate_name               TEXT         NOT NULL,
    substrate_ratio              TEXT         NOT NULL,
    bean_fraction                NUMERIC(6,4) NOT NULL
        CHECK (bean_fraction BETWEEN 0 AND 1),
    plantain_fraction            NUMERIC(6,4) NOT NULL
        CHECK (plantain_fraction BETWEEN 0 AND 1),
    slurry_level_code            TEXT,
    slurry_ratio                 TEXT         NOT NULL,
    -- Platform design space is 5-25 %TS; the published study spans 5.0-13.7.
    slurry_concentration_pct_ts  NUMERIC(6,2) NOT NULL
        CHECK (slurry_concentration_pct_ts BETWEEN 4.5 AND 25.5),
    carbon_nitrogen_ratio        NUMERIC(7,3) NOT NULL
        CHECK (carbon_nitrogen_ratio > 0),
    substrate_total_solids_pct   NUMERIC(6,2) NOT NULL,
    substrate_nitrogen_pct       NUMERIC(6,3) NOT NULL,
    substrate_ash_pct            NUMERIC(6,2) NOT NULL,
    working_volume_ml            NUMERIC(8,2) NOT NULL,
    is_reference_design          BOOLEAN      NOT NULL DEFAULT FALSE,
    beyond_published_envelope    BOOLEAN      NOT NULL DEFAULT FALSE,
    replicate_id                 SMALLINT     NOT NULL DEFAULT 1,
    CONSTRAINT reactor_config_blend_sums_to_one
        CHECK (abs(bean_fraction + plantain_fraction - 1) < 0.001)
);

COMMENT ON TABLE bioprocess.reactor_config IS
    'One row per bioreactor line. is_reference_design marks the 36 reactors '
    'replicating the published RCBD (DOI 10.30574/gscbps.2024.29.2.0423).';
COMMENT ON COLUMN bioprocess.reactor_config.beyond_published_envelope IS
    'TRUE where slurry %TS exceeds 13.72, the ceiling of the published design. '
    'Rows so flagged are extrapolation, not replication.';

-- --------------------------------------------------------------------------
-- Fact: one row per reactor-day
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bioprocess.reactor_telemetry (
    reading_id                  BIGSERIAL     PRIMARY KEY,
    reactor_id                  TEXT          NOT NULL
        REFERENCES bioprocess.reactor_config (reactor_id) ON DELETE CASCADE,
    reading_date                DATE          NOT NULL,
    day_index                   SMALLINT      NOT NULL
        CHECK (day_index BETWEEN 1 AND 60),
    daily_ph                    NUMERIC(4,2)  NOT NULL
        CHECK (daily_ph BETWEEN 3.5 AND 9.0),
    vfa_mg_l                    NUMERIC(10,2) NOT NULL CHECK (vfa_mg_l >= 0),
    alkalinity_mg_caco3_l       NUMERIC(10,2) NOT NULL CHECK (alkalinity_mg_caco3_l >= 0),
    temperature_c               NUMERIC(5,1)  NOT NULL
        CHECK (temperature_c BETWEEN 10 AND 60),
    daily_biogas_ml             NUMERIC(10,3) NOT NULL CHECK (daily_biogas_ml >= 0),
    cumulative_biogas_yield_ml  NUMERIC(12,3) NOT NULL
        CHECK (cumulative_biogas_yield_ml >= 0),
    specific_yield_ml_g_vs      NUMERIC(10,4) NOT NULL CHECK (specific_yield_ml_g_vs >= 0),
    methanogen_activity         NUMERIC(6,4)  NOT NULL
        CHECK (methanogen_activity BETWEEN 0 AND 1),
    process_state               TEXT          NOT NULL
        CHECK (process_state IN
               ('LAG','ACIDOGENIC','METHANOGENIC','SOURED','RECOVERING')),
    ingested_at                 TIMESTAMPTZ   NOT NULL DEFAULT now(),
    -- Makes the load idempotent: a re-triggered DAG run updates in place.
    CONSTRAINT reactor_telemetry_uniq UNIQUE (reactor_id, day_index)
);

COMMENT ON TABLE bioprocess.reactor_telemetry IS
    'Daily logging record per reactor. Replaces the manual spreadsheet '
    'transcription that bottlenecked the source study at 36 digesters.';

CREATE INDEX IF NOT EXISTS idx_telemetry_reactor  ON bioprocess.reactor_telemetry (reactor_id);
CREATE INDEX IF NOT EXISTS idx_telemetry_day      ON bioprocess.reactor_telemetry (day_index);
CREATE INDEX IF NOT EXISTS idx_telemetry_state    ON bioprocess.reactor_telemetry (process_state);
CREATE INDEX IF NOT EXISTS idx_telemetry_date     ON bioprocess.reactor_telemetry (reading_date);

-- --------------------------------------------------------------------------
-- Analytical views consumed by the RAG API's SQL retrieval arm
-- --------------------------------------------------------------------------

-- Final-day performance per reactor.
CREATE OR REPLACE VIEW bioprocess.v_reactor_performance AS
SELECT
    c.reactor_id,
    c.substrate_code,
    c.substrate_ratio,
    c.bean_fraction,
    c.slurry_ratio,
    c.slurry_concentration_pct_ts,
    c.carbon_nitrogen_ratio,
    c.is_reference_design,
    c.beyond_published_envelope,
    MAX(t.day_index)                      AS days_observed,
    MAX(t.cumulative_biogas_yield_ml)     AS final_cumulative_ml,
    AVG(t.daily_biogas_ml)                AS mean_daily_ml,
    MAX(t.daily_biogas_ml)                AS peak_daily_ml,
    MIN(t.daily_ph)                       AS min_ph,
    AVG(t.daily_ph)                       AS mean_ph,
    MAX(t.vfa_mg_l)                       AS peak_vfa_mg_l,
    COUNT(*) FILTER (WHERE t.process_state = 'SOURED')  AS soured_days,
    COUNT(*) FILTER (WHERE t.process_state = 'LAG')     AS lag_days
FROM bioprocess.reactor_config c
JOIN bioprocess.reactor_telemetry t USING (reactor_id)
GROUP BY c.reactor_id, c.substrate_code, c.substrate_ratio, c.bean_fraction,
         c.slurry_ratio, c.slurry_concentration_pct_ts, c.carbon_nitrogen_ratio,
         c.is_reference_design, c.beyond_published_envelope;

-- Reference cohort aggregated to the published Table 2 layout, so the API can
-- answer "does the platform reproduce the paper" without recomputation.
CREATE OR REPLACE VIEW bioprocess.v_reference_yield_matrix AS
SELECT
    c.substrate_code,
    c.slurry_level_code,
    c.slurry_ratio,
    ROUND(AVG(t.daily_biogas_ml), 3)  AS observed_mean_ml_day,
    COUNT(DISTINCT c.reactor_id)      AS replicates
FROM bioprocess.reactor_config c
JOIN bioprocess.reactor_telemetry t USING (reactor_id)
WHERE c.is_reference_design
GROUP BY c.substrate_code, c.slurry_level_code, c.slurry_ratio;

-- Reactors currently in or recovering from process failure.
CREATE OR REPLACE VIEW bioprocess.v_process_alerts AS
SELECT
    c.reactor_id,
    c.substrate_code,
    c.slurry_concentration_pct_ts,
    c.carbon_nitrogen_ratio,
    c.beyond_published_envelope,
    t.day_index,
    t.reading_date,
    t.daily_ph,
    t.vfa_mg_l,
    t.alkalinity_mg_caco3_l,
    ROUND(t.vfa_mg_l / NULLIF(t.alkalinity_mg_caco3_l, 0), 3) AS vfa_alkalinity_ratio,
    t.process_state
FROM bioprocess.reactor_config c
JOIN bioprocess.reactor_telemetry t USING (reactor_id)
WHERE t.process_state IN ('SOURED', 'RECOVERING')
   OR t.daily_ph < 6.5;
