"""Measured constants from the BioStreamer source study.

Single source of truth for every scientific value in the platform. The simulator,
the warehouse schema constraints, the load-time validation assertions and the RAG
corpus all import from here, so a transcription error can only be made once.

Source
------
Nnokwe JC, Orji MU, Ajuruchi VC, Jonas KC. "Effects of slurry concentration and
co-digestion on biogas yields from unseeded Phaseolus vulgaris (bean) peels chaff
and unseeded Musa paradisiaca (plantain) peels chaff."
GSC Biological and Pharmaceutical Sciences, 2024, 29(02), 214-218.
DOI: 10.30574/gscbps.2024.29.2.0423

Every value below carries a table reference. Do not edit without re-reading the
paper.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #

PAPER_DOI = "10.30574/gscbps.2024.29.2.0423"
PAPER_TITLE = (
    "Effects of slurry concentration and co-digestion on biogas yields from "
    "unseeded Phaseolus vulgaris (bean) peels chaff and unseeded Musa "
    "paradisiaca (plantain) peels chaff"
)
PAPER_AUTHORS = (
    "Nnokwe JC, Orji MU, Ajuruchi VC, Jonas KC"
)
PAPER_JOURNAL = "GSC Biological and Pharmaceutical Sciences"
PAPER_CITATION = f"{PAPER_AUTHORS} ({PAPER_JOURNAL}, 2024, 29(02), 214-218). DOI: {PAPER_DOI}"

# --------------------------------------------------------------------------- #
# Experimental design (paper, section 2.4 / 2.5)
# --------------------------------------------------------------------------- #

WORKING_VOLUME_ML = 753.0          # cm3 of slurry per 1000 ml mini-digester
DIGESTER_CAPACITY_ML = 1000.0
PAPER_RETENTION_DAYS = 37
PAPER_REPLICATES = 3
IS_SEEDED = False                  # unseeded - no cow rumen liquor. Drives long lags.


@dataclass(frozen=True)
class Substrate:
    """Proximate composition of one substrate (paper, Table 1)."""

    code: str
    name: str
    bean_fraction: float           # mass fraction of Phaseolus vulgaris in the blend
    ash_pct: float
    moisture_pct: float
    nitrogen_pct: float
    carbon_pct: float
    cn_ratio: float
    total_solids_pct: float
    substrate_ratio: str           # bean:plantain as reported


# Table 1. BP1u is a 0.691:1 bean:plantain blend -> bean fraction 0.691/1.691.
SUBSTRATES: dict[str, Substrate] = {
    "Bu": Substrate(
        code="Bu",
        name="Bean peel chaff (Phaseolus vulgaris)",
        bean_fraction=1.0,
        ash_pct=13.15,
        moisture_pct=3.97,
        nitrogen_pct=3.777,
        carbon_pct=34.15,
        cn_ratio=9.043,
        total_solids_pct=96.03,
        substrate_ratio="1:0",
    ),
    "Pu": Substrate(
        code="Pu",
        name="Plantain peel chaff (Musa paradisiaca)",
        bean_fraction=0.0,
        ash_pct=15.59,
        moisture_pct=9.88,
        nitrogen_pct=2.678,
        carbon_pct=37.89,
        cn_ratio=14.138,
        total_solids_pct=90.12,
        substrate_ratio="0:1",
    ),
    "BP1u": Substrate(
        code="BP1u",
        name="Mixed bean/plantain 1",
        bean_fraction=0.691 / 1.691,   # 0.40863...
        ash_pct=7.88,
        moisture_pct=15.85,
        nitrogen_pct=3.23,
        carbon_pct=44.73,
        cn_ratio=13.85,
        total_solids_pct=84.15,
        substrate_ratio="0.691:1",
    ),
    "BP2u": Substrate(
        code="BP2u",
        name="Mixed bean/plantain 2",
        bean_fraction=0.5,
        ash_pct=6.49,
        moisture_pct=15.06,
        nitrogen_pct=3.18,
        carbon_pct=48.65,
        cn_ratio=15.30,
        total_solids_pct=84.94,
        substrate_ratio="1:1",
    ),
}


@dataclass(frozen=True)
class SlurryLevel:
    """One slurry concentration level (paper, section 2.4)."""

    code: str
    water_parts: int               # parts water per 1 part substrate
    label: str

    @property
    def substrate_mass_fraction(self) -> float:
        """Fraction of total slurry mass that is substrate."""
        return 1.0 / (1.0 + self.water_parts)

    def total_solids_pct(self, substrate_ts_pct: float) -> float:
        """Slurry %TS given the substrate's own %TS.

        The paper reports concentration as a substrate:water mass ratio; the
        platform schema stores %TS, the standard AD process variable.
        """
        return self.substrate_mass_fraction * substrate_ts_pct


SLURRY_LEVELS: dict[str, SlurryLevel] = {
    "C1": SlurryLevel(code="C1", water_parts=6, label="1:6"),
    "C2": SlurryLevel(code="C2", water_parts=11, label="1:11"),
    "C3": SlurryLevel(code="C3", water_parts=16, label="1:16"),
}


# --------------------------------------------------------------------------- #
# Measured outcomes (paper, Table 2) - mean biogas yield in ml/day
# --------------------------------------------------------------------------- #

MEAN_DAILY_YIELD_ML: dict[tuple[str, str], float] = {
    ("Bu", "C1"): 4.40, ("Bu", "C2"): 5.06, ("Bu", "C3"): 0.34,
    ("Pu", "C1"): 0.55, ("Pu", "C2"): 0.00, ("Pu", "C3"): 0.00,
    ("BP1u", "C1"): 8.81, ("BP1u", "C2"): 1.30, ("BP1u", "C3"): 1.01,
    ("BP2u", "C1"): 3.69, ("BP2u", "C2"): 1.95, ("BP2u", "C3"): 1.86,
}

SUBSTRATE_TOTALS_ML_DAY = {"Bu": 9.80, "Pu": 0.55, "BP1u": 11.12, "BP2u": 7.50}
CONCENTRATION_TOTALS_ML_DAY = {"C1": 17.45, "C2": 8.31, "C3": 3.21}
GRAND_TOTAL_ML_DAY = 28.97

# Two-way ANOVA with replication (paper, Table 3).
ANOVA = {
    "substrate":     {"ss": 819.48,  "df": 3,   "ms": 273.16, "f": 11.93, "p": 1.61734e-07, "f_crit": 2.63},
    "slurry":        {"ss": 964.58,  "df": 2,   "ms": 482.29, "f": 21.06, "p": 1.87452e-09, "f_crit": 3.02},
    "interaction":   {"ss": 1053.58, "df": 6,   "ms": 175.60, "f": 7.67,  "p": 7.67804e-08, "f_crit": 2.12},
    "within":        {"ss": 9893.41, "df": 432, "ms": 22.90},
    "total":         {"ss": 12731.05, "df": 443},
}


# --------------------------------------------------------------------------- #
# Platform design space
# --------------------------------------------------------------------------- #

PLATFORM_REACTOR_COUNT = 100
PLATFORM_TIMELINE_DAYS = PAPER_RETENTION_DAYS  # match the paper's own 37-day window
TS_RANGE_PCT = (5.0, 25.0)

# The published design only reaches ~13.7 %TS (C1 with Bu). Anything above this is
# extrapolation and must be flagged wherever it surfaces.
PUBLISHED_TS_CEILING_PCT = 13.72

# Reference cohort: 4 substrates x 3 concentrations x 3 replicates = 36 reactors.
REFERENCE_REACTOR_COUNT = len(SUBSTRATES) * len(SLURRY_LEVELS) * PAPER_REPLICATES
EXPLORATORY_REACTOR_COUNT = PLATFORM_REACTOR_COUNT - REFERENCE_REACTOR_COUNT

PROCESS_STATES = ("LAG", "ACIDOGENIC", "METHANOGENIC", "SOURED", "RECOVERING")


# --------------------------------------------------------------------------- #
# Process-chemistry constants (literature-general, NOT from the source paper)
# --------------------------------------------------------------------------- #
# The source study reports no pH or temperature series. The values below come from
# standard anaerobic-digestion literature and are used to give the simulation a
# defensible mechanism. They are tagged so the RAG layer can distinguish a
# published finding from modelled context.

PH_INITIAL_RANGE = (6.7, 7.3)          # freshly prepared unseeded slurry
PH_METHANOGEN_OPTIMUM = 7.0
PH_INHIBITION_ONSET = 6.5              # methanogenesis begins to slow
PH_SEVERE_INHIBITION = 6.3             # strongly suppressed
PH_SOURED_THRESHOLD = 6.0              # process failure ("souring")
PH_FLOOR = 4.8                         # VFA-saturated floor

TEMPERATURE_MEAN_C = 30.5              # ambient mesophilic, Owerri, Imo State
TEMPERATURE_SD_C = 1.4

VFA_INITIAL_MG_L = 250.0
# Alkalinity scales with substrate nitrogen (ammonia buffering). A C/N of ~9
# (bean) buffers better than ~15 (BP2u).
ALKALINITY_PER_N_PCT = 620.0           # mg CaCO3/L per % substrate N
ALKALINITY_FLOOR_MG_L = 400.0

# Volatile solids as a fraction of total solids, from ash content: VS = TS - ash.
def volatile_solids_fraction(substrate_ash_pct: float) -> float:
    """VS as a fraction of TS, derived from measured ash (Table 1)."""
    return max(0.0, 1.0 - substrate_ash_pct / 100.0)


# --------------------------------------------------------------------------- #
# Interpolation across measured anchor points
# --------------------------------------------------------------------------- #

def _anchors() -> list[Substrate]:
    """Substrates sorted by bean fraction, for piecewise-linear interpolation."""
    return sorted(SUBSTRATES.values(), key=lambda s: s.bean_fraction)


def interpolate_substrate(bean_fraction: float) -> dict[str, float]:
    """Proximate composition for an arbitrary bean:plantain blend.

    Piecewise-linear across the four *measured* anchor points rather than an
    assumed linear mixing law, so the interpolant passes exactly through the
    paper's Table 1 values -- including the non-monotonic carbon content, which a
    naive linear mix would smooth away.
    """
    bf = min(max(bean_fraction, 0.0), 1.0)
    anchors = _anchors()

    lo, hi = anchors[0], anchors[-1]
    for left, right in zip(anchors, anchors[1:]):
        if left.bean_fraction <= bf <= right.bean_fraction:
            lo, hi = left, right
            break

    span = hi.bean_fraction - lo.bean_fraction
    w = 0.0 if span == 0 else (bf - lo.bean_fraction) / span

    def mix(attr: str) -> float:
        return getattr(lo, attr) * (1 - w) + getattr(hi, attr) * w

    return {
        "ash_pct": mix("ash_pct"),
        "moisture_pct": mix("moisture_pct"),
        "nitrogen_pct": mix("nitrogen_pct"),
        "carbon_pct": mix("carbon_pct"),
        "cn_ratio": mix("cn_ratio"),
        "total_solids_pct": mix("total_solids_pct"),
    }


def interpolate_substrate_yield_potential(bean_fraction: float) -> float:
    """Relative yield potential (ml/day summed over the 3 concentrations).

    Interpolates the paper's Table 2 row totals across bean fraction. The result
    is deliberately non-monotonic: it peaks at BP1u (bean fraction 0.409), which
    is the study's headline finding -- co-digestion at 0.691:1 outperforms both
    equal-proportion co-digestion and either single substrate.
    """
    bf = min(max(bean_fraction, 0.0), 1.0)
    pts = sorted(
        (s.bean_fraction, SUBSTRATE_TOTALS_ML_DAY[s.code]) for s in SUBSTRATES.values()
    )

    if bf <= pts[0][0]:
        return pts[0][1]
    if bf >= pts[-1][0]:
        return pts[-1][1]

    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= bf <= x1:
            w = (bf - x0) / (x1 - x0) if x1 != x0 else 0.0
            return y0 * (1 - w) + y1 * w
    return pts[-1][1]


def concentration_response(ts_pct: float) -> float:
    """Relative yield multiplier as a function of slurry %TS.

    Anchored on the paper's concentration totals (C1 17.45 > C2 8.31 > C3 3.21
    ml/day) which establish a strongly increasing response across 5.6-13.7 %TS.
    Above the published ceiling the curve rolls off to represent the
    well-documented VFA-accumulation and ammonia-inhibition regime of high-solids
    ("dry") digestion. That roll-off is an extrapolation beyond the source study
    and is flagged as such wherever the data surfaces.
    """
    bu_ts = SUBSTRATES["Bu"].total_solids_pct
    anchors = sorted(
        (SLURRY_LEVELS[c].total_solids_pct(bu_ts), CONCENTRATION_TOTALS_ML_DAY[c])
        for c in ("C1", "C2", "C3")
    )
    # Normalise so C1 (the paper's best level) == 1.0
    peak = max(v for _, v in anchors)

    ts = min(max(ts_pct, TS_RANGE_PCT[0]), TS_RANGE_PCT[1])

    if ts <= anchors[0][0]:
        return anchors[0][1] / peak
    if ts <= anchors[-1][0]:
        for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
            if x0 <= ts <= x1:
                w = (ts - x0) / (x1 - x0)
                return (y0 * (1 - w) + y1 * w) / peak

    # Beyond the published envelope: plateau to ~16 %TS, then inhibition roll-off.
    over = ts - anchors[-1][0]
    if over <= 2.5:
        return 1.0 + 0.04 * over          # mild continued gain
    decay = (over - 2.5) / (TS_RANGE_PCT[1] - anchors[-1][0] - 2.5)
    return max(0.18, 1.10 * (1.0 - 0.82 * decay ** 1.35))
