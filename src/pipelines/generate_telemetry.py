"""Track 1 — structured telemetry simulation for 100 parallel bioreactor lines.

Generates a scientifically grounded daily-logging dataset for 100 anaerobic
digesters over a 30-day timeline, mirroring the parameters of:

    Nnokwe JC, Orji MU, Ajuruchi VC, Jonas KC. "Effects of slurry concentration
    and co-digestion on biogas yields from unseeded Phaseolus vulgaris (bean)
    peels chaff and unseeded Musa paradisiaca (plantain) peels chaff."
    GSC Biol. Pharm. Sci., 2024, 29(02), 214-218.
    DOI: 10.30574/gscbps.2024.29.2.0423

Design
------
This is not noise applied to a fitted curve. Each reactor-day runs a coupled
acidogenesis/methanogenesis state model:

    hydrolysis -> VFA accumulation
    substrate N -> alkalinity (ammonia buffering)
    VFA : alkalinity -> pH
    pH -> methanogen activity gate      <-- the causal link
    methanogen activity -> daily biogas

The *shape* comes from the mechanism; the *magnitude* is calibrated per
reactor so the 36 reference reactors reproduce the paper's Table 2 means. That
split is deliberate: mechanism where we can defend it, calibration where the
paper gives us ground truth.

Reactor allocation
------------------
  36 reference reactors  - exact replication of the published RCBD
                           (4 substrates x 3 concentrations x 3 replicates)
  64 exploratory reactors - continuous bean fraction x 5-25 %TS, sampling the
                           design space the original study could not afford

Determinism
-----------
One seeded numpy Generator drives the entire run. Same seed => identical dataset.

Usage
-----
    python -m src.pipelines.generate_telemetry --out data/telemetry.parquet
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import science as sci
from src.common.config import get_settings

LOG = logging.getLogger("biostreamer.telemetry")

E = math.e

# Ceiling on the post-hoc magnitude calibration (see simulate_reactor, pass 2).
# A healthy reactor needs ~1.5-2x; anything requiring more than this was driven
# to failure by the mechanism and must stay suppressed rather than be inflated
# back to its nominal target.
MAX_CALIBRATION_SCALE = 4.0

# Acid production per unit substrate load, mg/L/day at 10 %TS. Calibrated so a
# healthy reactor in the published envelope settles near pH 6.6-7.0 while the
# low-nitrogen substrate (Pu, C/N 14.1) drifts into the inhibition band -- which
# is the mechanism behind its near-zero measured yield.
VFA_PRODUCTION_COEFF = 72.0

# Logistic establishment rate of the methanogenic consortium.
METHANOGEN_GROWTH_RATE = 0.38


# --------------------------------------------------------------------------- #
# Reactor configuration
# --------------------------------------------------------------------------- #

def build_reactor_configs(rng: np.random.Generator, n_reactors: int) -> pd.DataFrame:
    """Assemble the 100-reactor design: 36 reference + 64 exploratory."""
    rows: list[dict] = []
    idx = 1

    # --- Reference cohort: the published RCBD, in triplicate ------------------
    for code in ("Bu", "Pu", "BP1u", "BP2u"):
        sub = sci.SUBSTRATES[code]
        for conc_code in ("C1", "C2", "C3"):
            level = sci.SLURRY_LEVELS[conc_code]
            ts_pct = level.total_solids_pct(sub.total_solids_pct)
            for rep in range(1, sci.PAPER_REPLICATES + 1):
                rows.append(
                    {
                        "reactor_id": f"R{idx:03d}",
                        "substrate_code": sub.code,
                        "substrate_name": sub.name,
                        "substrate_ratio": sub.substrate_ratio,
                        "bean_fraction": round(sub.bean_fraction, 4),
                        "plantain_fraction": round(1.0 - sub.bean_fraction, 4),
                        "slurry_level_code": conc_code,
                        "slurry_ratio": level.label,
                        "slurry_concentration_pct_ts": round(ts_pct, 2),
                        "carbon_nitrogen_ratio": round(sub.cn_ratio, 3),
                        "substrate_total_solids_pct": round(sub.total_solids_pct, 2),
                        "substrate_nitrogen_pct": round(sub.nitrogen_pct, 3),
                        "substrate_ash_pct": round(sub.ash_pct, 2),
                        "working_volume_ml": sci.WORKING_VOLUME_ML,
                        "is_reference_design": True,
                        "replicate_id": rep,
                        # Calibration target straight from Table 2.
                        "_target_mean_ml_day": sci.MEAN_DAILY_YIELD_ML[(code, conc_code)],
                    }
                )
                idx += 1

    # --- Exploratory cohort: continuous design space -------------------------
    n_explore = max(0, n_reactors - len(rows))
    for _ in range(n_explore):
        bean_fraction = float(rng.uniform(0.0, 1.0))
        props = sci.interpolate_substrate(bean_fraction)
        ts_pct = float(rng.uniform(*sci.TS_RANGE_PCT))

        # Modelled target: substrate potential x concentration response, spread
        # across the three-level total the paper reports as a row sum.
        potential = sci.interpolate_substrate_yield_potential(bean_fraction) / 3.0
        target = potential * sci.concentration_response(ts_pct)

        # Implied water:substrate ratio for this %TS, for traceability.
        mass_fraction = ts_pct / props["total_solids_pct"]
        water_parts = (1.0 / mass_fraction) - 1.0 if mass_fraction > 0 else float("inf")

        rows.append(
            {
                "reactor_id": f"R{idx:03d}",
                "substrate_code": "BLEND",
                "substrate_name": "Exploratory bean/plantain blend",
                "substrate_ratio": f"{bean_fraction:.3f}:{1 - bean_fraction:.3f}",
                "bean_fraction": round(bean_fraction, 4),
                "plantain_fraction": round(1.0 - bean_fraction, 4),
                "slurry_level_code": None,
                "slurry_ratio": f"1:{water_parts:.1f}",
                "slurry_concentration_pct_ts": round(ts_pct, 2),
                "carbon_nitrogen_ratio": round(props["cn_ratio"], 3),
                "substrate_total_solids_pct": round(props["total_solids_pct"], 2),
                "substrate_nitrogen_pct": round(props["nitrogen_pct"], 3),
                "substrate_ash_pct": round(props["ash_pct"], 2),
                "working_volume_ml": sci.WORKING_VOLUME_ML,
                "is_reference_design": False,
                "replicate_id": 1,
                "_target_mean_ml_day": round(target, 4),
            }
        )
        idx += 1

    df = pd.DataFrame(rows)
    df["beyond_published_envelope"] = (
        df["slurry_concentration_pct_ts"] > sci.PUBLISHED_TS_CEILING_PCT
    )
    return df


# --------------------------------------------------------------------------- #
# Kinetics
# --------------------------------------------------------------------------- #

def gompertz_cumulative(t: np.ndarray, p: float, rm: float, lag: float) -> np.ndarray:
    """Modified Gompertz cumulative biogas model.

        M(t) = P * exp{ -exp[ (Rm * e / P)(lambda - t) + 1 ] }

    Standard in AD kinetics; cf. Latinwo & Agarry and Olugbemide et al., both
    cited by the source paper.
    """
    if p <= 0 or rm <= 0:
        return np.zeros_like(t, dtype=float)
    exponent = (rm * E / p) * (lag - t) + 1.0
    exponent = np.clip(exponent, -700, 700)          # guard float overflow
    return p * np.exp(-np.exp(exponent))


def _lag_phase_days(rng: np.random.Generator, cn_ratio: float, ts_pct: float) -> float:
    """Lag phase for an *unseeded* digester.

    With no inoculum the methanogenic consortium must establish from the native
    microflora of the peels, so lags run 8-22 d rather than the 2-4 d typical of
    seeded systems. This is what reproduces Pu's near-zero yields over a short
    retention window -- its lag routinely exceeds the observation period.
    """
    base = 11.0
    # Low C/N -> more readily available N -> faster establishment.
    base += (cn_ratio - 12.0) * 0.45
    # High solids -> mass-transfer limited, slower start.
    base += max(0.0, ts_pct - sci.PUBLISHED_TS_CEILING_PCT) * 0.55
    return float(np.clip(rng.normal(base, 2.6), 3.0, 28.0))


# --------------------------------------------------------------------------- #
# Per-reactor simulation
# --------------------------------------------------------------------------- #

def simulate_reactor(
    cfg: pd.Series,
    rng: np.random.Generator,
    n_days: int,
    start_date: date,
) -> list[dict]:
    """Run the coupled VFA/pH/methanogen model for one reactor."""
    ts_pct = float(cfg["slurry_concentration_pct_ts"])
    cn_ratio = float(cfg["carbon_nitrogen_ratio"])
    n_pct = float(cfg["substrate_nitrogen_pct"])
    ash_pct = float(cfg["substrate_ash_pct"])
    target_mean = float(cfg["_target_mean_ml_day"])

    vs_fraction = sci.volatile_solids_fraction(ash_pct)
    # Grams of volatile solids in the digester.
    vs_grams = sci.WORKING_VOLUME_ML * (ts_pct / 100.0) * vs_fraction

    # --- Kinetic envelope ----------------------------------------------------
    lag = _lag_phase_days(rng, cn_ratio, ts_pct)
    days = np.arange(1, n_days + 1, dtype=float)

    # Choose P so that mean daily output over the window matches the target.
    # Solve for the Gompertz shape factor at the horizon, then scale.
    provisional_p = max(target_mean * n_days * 3.0, 1e-6)
    provisional_rm = max(target_mean * 2.2, 1e-6)
    shape = gompertz_cumulative(days, provisional_p, provisional_rm, lag)[-1] / provisional_p
    shape = max(shape, 1e-4)

    ultimate_p = (target_mean * n_days) / shape
    peak_rate = max(target_mean * 2.2, 1e-6)

    envelope = gompertz_cumulative(days, ultimate_p, peak_rate, lag)
    daily_potential = np.diff(np.concatenate([[0.0], envelope]))
    daily_potential = np.clip(daily_potential, 0.0, None)

    # --- Buffering capacity --------------------------------------------------
    # Alkalinity scales with the *nitrogen load*, which is proportional to the
    # mass of substrate charged -- i.e. linearly with %TS. This is what keeps the
    # VFA:alkalinity ratio roughly TS-invariant inside the published envelope,
    # and is why the paper observes higher yields at higher concentration rather
    # than the souring a naive "more solids => more acid" model would predict.
    alkalinity = max(
        sci.ALKALINITY_FLOOR_MG_L,
        n_pct * sci.ALKALINITY_PER_N_PCT * (ts_pct / 10.0),
    )
    alkalinity *= float(rng.normal(1.0, 0.06))

    # Above the published ceiling, mass-transfer limitation and ammonia
    # inhibition break that balance: acidification outruns buffering. This term
    # is an extrapolation beyond the source study (see science.py).
    overload = 1.0 + 0.155 * max(0.0, ts_pct - sci.PUBLISHED_TS_CEILING_PCT) ** 1.45

    # --- State ---------------------------------------------------------------
    ph = float(rng.uniform(*sci.PH_INITIAL_RANGE))
    vfa = sci.VFA_INITIAL_MG_L * (ts_pct / 10.0)
    methanogen_activity = 0.03          # near-zero: unseeded
    cumulative = 0.0
    soured_seen = False

    # Pass 1 accumulates the mechanistic modulation; magnitude is calibrated
    # afterwards (see below), so the loop below records state, not final yield.
    staged: list[dict] = []
    modulation = np.zeros(n_days, dtype=float)

    for i, day in enumerate(range(1, n_days + 1)):
        # 1. Hydrolysis/acidogenesis: VFA production scales with available solids
        #    and decays as the readily degradable fraction is consumed.
        hydrolysis_decay = math.exp(-0.055 * day)
        vfa_produced = VFA_PRODUCTION_COEFF * (ts_pct / 10.0) * hydrolysis_decay * overload
        vfa_produced *= float(rng.normal(1.0, 0.10))

        # 2. Methanogenic consumption of VFA, proportional to active biomass.
        #    Also scales with load: a bigger population processes more acid.
        vfa_consumed = 430.0 * methanogen_activity * (ts_pct / 10.0)
        vfa = max(60.0, vfa + vfa_produced - vfa_consumed)

        # 3. pH from the VFA : alkalinity ratio. Not sampled independently --
        #    this is what makes pH and gas production causally linked.
        ratio = vfa / alkalinity
        ph_target = sci.PH_METHANOGEN_OPTIMUM - 1.75 * math.log1p(max(0.0, ratio - 0.25))
        ph_target = float(np.clip(ph_target, sci.PH_FLOOR, 7.6))
        ph += (ph_target - ph) * 0.42 + float(rng.normal(0.0, 0.045))
        ph = float(np.clip(ph, sci.PH_FLOOR, 8.2))

        # 4. pH gate on methanogenesis.
        if ph >= sci.PH_INHIBITION_ONSET:
            gate = 1.0
        elif ph >= sci.PH_SEVERE_INHIBITION:
            gate = 0.45
        elif ph >= sci.PH_SOURED_THRESHOLD:
            gate = 0.12
        else:
            gate = 0.0

        # 5. Logistic growth of the methanogen population, gated by pH.
        growth = METHANOGEN_GROWTH_RATE * methanogen_activity * (1.0 - methanogen_activity) * gate
        decay = 0.06 * methanogen_activity * (1.0 - gate)
        methanogen_activity = float(
            np.clip(methanogen_activity + growth - decay, 0.0, 1.0)
        )

        # 6. Mechanistic modulation of the kinetic potential by the live
        #    population. Magnitude is calibrated after the loop.
        activity_factor = 0.25 + 0.75 * methanogen_activity
        modulation[i] = gate * activity_factor * float(rng.normal(1.0, 0.13))

        # 7. Process state label.
        if ph < sci.PH_SOURED_THRESHOLD:
            state = "SOURED"
            soured_seen = True
        elif soured_seen and ph >= sci.PH_INHIBITION_ONSET:
            state = "RECOVERING"
        elif day < lag:
            state = "LAG"
        elif methanogen_activity < 0.20:
            state = "ACIDOGENIC"
        else:
            state = "METHANOGENIC"

        staged.append(
            {
                "reactor_id": cfg["reactor_id"],
                "reading_date": start_date + timedelta(days=day - 1),
                "day_index": day,
                "daily_ph": round(ph, 2),
                "vfa_mg_l": round(vfa, 2),
                "alkalinity_mg_caco3_l": round(alkalinity, 2),
                "temperature_c": round(
                    float(rng.normal(sci.TEMPERATURE_MEAN_C, sci.TEMPERATURE_SD_C)), 1
                ),
                "methanogen_activity": round(methanogen_activity, 4),
                "process_state": state,
            }
        )

    # --- Pass 2: calibrate magnitude ----------------------------------------
    # The mechanism above sets the *shape* (when gas appears, how a pH crash
    # suppresses it); the paper's Table 2 sets the *magnitude*. Scaling the raw
    # series to the target mean corrects the systematic undershoot introduced by
    # the modulation term without discarding the mechanism.
    #
    # The scale is capped: a reactor the mechanism drove to failure must stay
    # suppressed rather than being inflated back up to its nominal target. That
    # cap is what preserves the study's real finding that some unseeded
    # digesters (notably Pu at C2/C3) simply never produce.
    raw = np.clip(daily_potential * np.clip(modulation, 0.0, None), 0.0, None)
    raw_total = float(raw.sum())
    if raw_total > 1e-9 and target_mean > 0:
        scale = min((target_mean * n_days) / raw_total, MAX_CALIBRATION_SCALE)
    else:
        scale = 0.0

    records: list[dict] = []
    cumulative = 0.0
    for i, row in enumerate(staged):
        # Water-displacement readings are taken to the nearest 0.5 ml.
        daily = round(float(raw[i] * scale) * 2.0) / 2.0
        daily = max(0.0, daily)
        cumulative += daily
        row["daily_biogas_ml"] = round(daily, 3)
        row["cumulative_biogas_yield_ml"] = round(cumulative, 3)
        row["specific_yield_ml_g_vs"] = round(
            cumulative / vs_grams if vs_grams > 0 else 0.0, 4
        )
        records.append(row)

    return records


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def generate(
    seed: int | None = None,
    n_reactors: int | None = None,
    n_days: int | None = None,
    start_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate (reactor_config, reactor_telemetry) as a pair of DataFrames."""
    s = get_settings().simulation
    seed = seed if seed is not None else s.seed
    n_reactors = n_reactors or s.reactor_count
    n_days = n_days or s.timeline_days
    start = datetime.strptime(start_date or s.start_date, "%Y-%m-%d").date()

    rng = np.random.default_rng(seed)

    LOG.info(
        "Simulating %d reactors x %d days (seed=%d, start=%s)",
        n_reactors, n_days, seed, start,
    )

    configs = build_reactor_configs(rng, n_reactors)

    telemetry_rows: list[dict] = []
    for _, cfg in configs.iterrows():
        telemetry_rows.extend(simulate_reactor(cfg, rng, n_days, start))

    telemetry = pd.DataFrame(telemetry_rows)
    telemetry["ingested_at"] = datetime.now(timezone.utc)

    # Drop the calibration target from the persisted config -- it is an internal
    # simulation input, not an observation.
    configs_out = configs.drop(columns=["_target_mean_ml_day"])

    LOG.info(
        "Generated %d telemetry rows across %d reactors",
        len(telemetry), configs_out.shape[0],
    )
    return configs_out, telemetry


def validate_against_paper(
    configs: pd.DataFrame, telemetry: pd.DataFrame, tolerance: float = 0.25
) -> pd.DataFrame:
    """Assert the reference cohort reproduces the paper's Table 2 means.

    Tolerance is relative. Cells whose published mean is 0.00 ml/day (Pu at C2
    and C3) are checked against an absolute ceiling instead, since a relative
    tolerance is undefined there.
    """
    ref = configs[configs["is_reference_design"]]
    merged = telemetry.merge(
        ref[["reactor_id", "substrate_code", "slurry_level_code"]],
        on="reactor_id",
        how="inner",
    )

    observed = (
        merged.groupby(["substrate_code", "slurry_level_code"])["daily_biogas_ml"]
        .mean()
        .reset_index(name="observed_ml_day")
    )
    observed["published_ml_day"] = observed.apply(
        lambda r: sci.MEAN_DAILY_YIELD_ML[
            (r["substrate_code"], r["slurry_level_code"])
        ],
        axis=1,
    )

    def _ok(row) -> bool:
        pub, obs = row["published_ml_day"], row["observed_ml_day"]
        if pub == 0.0:
            return obs <= 0.35
        return abs(obs - pub) / pub <= tolerance

    observed["within_tolerance"] = observed.apply(_ok, axis=1)
    observed["rel_error"] = observed.apply(
        lambda r: None
        if r["published_ml_day"] == 0
        else round((r["observed_ml_day"] - r["published_ml_day"]) / r["published_ml_day"], 4),
        axis=1,
    )
    return observed


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    ap = argparse.ArgumentParser(description="Generate BioStreamer reactor telemetry.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--reactors", type=int, default=None)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--start-date", type=str, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("data/generated"))
    ap.add_argument("--validate", action="store_true", help="Check against paper Table 2")
    args = ap.parse_args()

    configs, telemetry = generate(
        seed=args.seed,
        n_reactors=args.reactors,
        n_days=args.days,
        start_date=args.start_date,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = args.out_dir / "reactor_config.parquet"
    tel_path = args.out_dir / "reactor_telemetry.parquet"
    configs.to_parquet(cfg_path, index=False)
    telemetry.to_parquet(tel_path, index=False)
    LOG.info("Wrote %s and %s", cfg_path, tel_path)

    if args.validate:
        report = validate_against_paper(configs, telemetry)
        print("\nReference cohort vs published Table 2 (ml/day):")
        print(report.to_string(index=False))
        failures = report[~report["within_tolerance"]]
        if len(failures):
            LOG.error("%d cell(s) outside tolerance", len(failures))
            return 1
        LOG.info("All 12 reference cells reproduce the published means.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
