"""Literature corpus for the Track 2 (unstructured) knowledge pipeline.

Two provenance classes, and the distinction is load-bearing for the RAG layer:

    "published_finding" - text drawn directly from the source paper. A claim
        retrieved from one of these passages is attributable to a specific
        section of a specific peer-reviewed article and can be cited as such.

    "domain_context"     - general anaerobic-digestion process knowledge, not
        reported in the source paper (it measured no pH or temperature
        series). Included so the assistant can explain *mechanism* --
        why VFA accumulation drops pH, why that inhibits methanogens --
        without those explanations being mistaken for a measured result.

Every entry keeps this tag through chunking and into the Qdrant payload, so a
downstream consumer can always tell which kind of context it is looking at.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common.science import PAPER_CITATION, PAPER_DOI, PAPER_TITLE


@dataclass(frozen=True)
class CorpusDocument:
    text: str
    section: str
    provenance: str            # "published_finding" | "domain_context"
    source: str = field(default=PAPER_TITLE)
    doi: str | None = field(default=None)


def _paper(text: str, section: str) -> CorpusDocument:
    return CorpusDocument(
        text=text.strip(), section=section, provenance="published_finding", doi=PAPER_DOI
    )


def _context(text: str, section: str, source: str) -> CorpusDocument:
    return CorpusDocument(
        text=text.strip(), section=section, provenance="domain_context", source=source, doi=None
    )


# --------------------------------------------------------------------------- #
# Published findings -- verbatim / faithfully condensed from the source paper
# --------------------------------------------------------------------------- #

_PAPER_DOCS: list[CorpusDocument] = [
    _paper(
        f"""
        {PAPER_TITLE}. {PAPER_CITATION}

        Abstract. This study explored converting agricultural wastes of bean peel
        chaff (Bu) and plantain peel chaff (Pu) into biogas, an eco-friendly
        biofuel, without cow rumen liquor. By digesting these wastes anaerobically
        in various combinations and slurry concentrations, the study assessed
        their individual and combined biogas yields. Bean and plantain peels were
        digested alone and in two mixtures: BP1u (bean-to-plantain ratio 0.691:1)
        and BP2u (1:1 ratio). Three waste-to-water slurry concentrations (1:6,
        1:11, and 1:16) were used, with anaerobic digestion lasting 37 days in
        triplicate 1000 ml mini digesters. Biogas production was measured via
        water displacement.
        """,
        "Abstract",
    ),
    _paper(
        """
        The study found total solids percentages of 96.03% for Bu, 90.12% for Pu,
        84.15% for BP1u, and 84.94% for BP2u. The carbon/nitrogen (C/N) ratios
        ranged from 9.04 to 15.3. Average biogas yields showed BP1u produced the
        highest (11.12 ml/day), followed by Bu (9.8 ml/day), BP2u (7.5 ml/day),
        and Pu (0.55 ml/day). Among slurry concentrations, 1:6 (C1) yielded the
        most biogas (17.45 ml/day), followed by 1:11 (C2) and 1:16 (C3). ANOVA
        indicated significant effects from both slurry concentration and
        co-digestion on biogas yield, along with a notable interaction between
        both factors.
        """,
        "Abstract",
    ),
    _paper(
        """
        In conclusion, both bean and plantain peels can generate biogas
        effectively, especially when combined and digested in higher slurry
        concentrations for improved yield. This approach supports waste
        management and renewable energy goals by transforming agricultural
        residues into valuable biofuel. Keywords: Bean peels; Plantain peels;
        Slurry concentration; Co-digestion; Biogas.
        """,
        "Abstract",
    ),
    _paper(
        """
        Introduction. Conventional energy sources such as fossil fuels, coal,
        oil and gas are being depleted and constitute an environmental concern
        through their contribution to global warming; per the UN, fossil fuels
        account for at least 75% of all greenhouse gas emissions and 90% of
        carbon dioxide emissions. Renewable energy sources -- the sun, wind,
        water, waste, tides, waves, and geothermal heat -- are comparatively
        abundant, non-harmful, and continuously replenished. Among these, energy
        from wastes has been widely studied, and several waste-to-wealth
        technologies converting waste into biodiesel and biogas are already in
        place. The anaerobic digestion of wastes has received special attention
        because, beyond biogas production, the resulting digestate can improve
        soil fertility.
        """,
        "Introduction",
    ),
    _paper(
        """
        Greater attention has historically been given to animal manure and
        sewage sludge, digested singly or in combination, as sources of biogas.
        This underscores the need to direct attention to agricultural wastes,
        food waste, and kitchen wastes, which are more abundant, especially in
        urban areas, where they constitute an environmental hazard if not
        efficiently managed. Among these agro-wastes are food and fruit peels.
        """,
        "Introduction",
    ),
    _paper(
        """
        Phaseolus vulgaris (beans) is a staple food in tropical regions and a
        legume that thrives in fertile soils, serving as an essential protein
        source. In Southeast Nigeria, two popular bean dishes are fried bean
        cakes ("akara") and steamed bean cakes ("moi-moi"); preparing these
        dishes removes the outer seed coat, which is sometimes repurposed as
        animal feed but often becomes waste in urban areas. Musa paradisiaca
        (plantain) is extensively grown and consumed throughout Nigeria, commonly
        enjoyed as roasted unripe fruit or fried in both ripe and unripe forms.
        While plantain peels can serve as animal feed or natural fertilizer, in
        urban areas without livestock or farms these peels often become waste.
        The study's aim was to evaluate the conversion of bean peels and plantain
        peels into biogas without the conventional addition of an inoculum, while
        studying the effects of slurry concentration and co-digestion.
        """,
        "Introduction",
    ),
    _paper(
        """
        Materials and methods -- Collection and preparation of substrates. Fresh
        peels of Phaseolus vulgaris (beans) and Musa paradisiaca (plantain) were
        obtained from food vendors along Ikenegbu Road and Okigwe Road in Owerri,
        the capital of Imo State, Nigeria. The collected peels were air-dried,
        ground into powder, stored in sterile containers, and labeled Bu for bean
        peels and Pu for plantain peels.
        """,
        "Methods",
    ),
    _paper(
        """
        Proximate analysis. Samples underwent proximate analysis to measure
        moisture content, total solids, protein and carbohydrate levels, and the
        carbon-nitrogen (C/N) ratio, following the 2003 guidelines of the
        Association of Official Agricultural Chemists (AOAC).
        """,
        "Methods",
    ),
    _paper(
        """
        Preparation of peel mixtures. A 1:1 mass ratio of Bu and Pu was combined
        to create BP2u. Based on the C/N ratios and moisture contents of Bu and
        Pu, an intermediate C/N ratio was calculated to form sample BP1u, a
        bean-to-plantain ratio of 0.691:1. All four samples -- Bu, Pu, BP1u, and
        BP2u -- were stored in pre-labeled sterile containers.
        """,
        "Methods",
    ),
    _paper(
        """
        Preparation of slurries. Mini digesters of 1000 ml capacity were cleaned,
        dried, and sterilized at 121 degrees C for 15 minutes before anaerobic
        digestion. Each digester was sealed, fitted with a gas outlet PVC hose
        leading to an inverted, water-filled, calibrated plastic cylinder for gas
        collection and daily measurement -- the water displacement method. Each
        digester held 753 cm3 of slurry. Three slurry concentrations of the
        substrates -- 1:6 (C1), 1:11 (C2), and 1:16 (C3) substrate-to-water
        ratios -- were prepared in triplicate and transferred aseptically into
        each digester. Biogas was measured daily over a 37-day retention period,
        with readings recorded regularly. No cow rumen liquor or other inoculum
        was added -- the digesters were unseeded.
        """,
        "Methods",
    ),
    _paper(
        """
        Experimental set-up and design. The randomized complete block design
        (RCBD) was used for the experiment. There were two factors: digestion
        mode/type (four levels: Bu, Pu, BP1u, and BP2u) and slurry concentration
        (three levels: C1, C2, and C3), with randomization.

        Analysis of results. The daily biogas readings were recorded and later
        analysed using Microsoft Excel 365 Data Analysis Toolpak (two-way ANOVA
        with repetition).
        """,
        "Methods",
    ),
    _paper(
        """
        Results -- Proximate composition (Table 1). Bean peel chaff (Bu): ash
        content 13.15%, moisture content 3.97%, nitrogen 3.777%, carbon 34.15%,
        C/N ratio 9.043, total solids 96.03%. Plantain peel chaff (Pu): ash
        15.59%, moisture 9.88%, nitrogen 2.678%, carbon 37.89%, C/N ratio 14.138,
        total solids 90.12%. Mixed Bean/Plantain 1 (BP1u): ash 7.88%, moisture
        15.85%, nitrogen 3.23%, carbon 44.73%, C/N ratio 13.85, total solids
        84.15%. Mixed Bean/Plantain 2 (BP2u): ash 6.49%, moisture 15.06%,
        nitrogen 3.18%, carbon 48.65%, C/N ratio 15.3, total solids 84.94%.

        These findings align with the view that agricultural waste typically has
        high organic content and a C/N ratio within the optimal range, and that
        high total solids facilitate methanization. Notably, the C/N ratios of
        Bu, Pu, and BP1u were below 15; despite this, the substrates generated
        considerable amounts of biogas.
        """,
        "Results",
    ),
    _paper(
        """
        Results -- Biogas yields of Bu, Pu, BP1u, and BP2u (Table 2, ml/day).
        Bu: C1 (1:6) = 4.40, C2 (1:11) = 5.06, C3 (1:16) = 0.34, total = 9.80.
        Pu: C1 = 0.55, C2 = 0.00, C3 = 0.00, total = 0.55.
        BP1u: C1 = 8.81, C2 = 1.30, C3 = 1.01, total = 11.12.
        BP2u: C1 = 3.69, C2 = 1.95, C3 = 1.86, total = 7.50.
        Column totals across substrates: C1 = 17.45, C2 = 8.31, C3 = 3.21,
        grand total = 28.97 ml/day.

        The substrates yielded varying amounts of biogas, with BP1u showing the
        highest average yield and Pu the lowest. Co-digesting bean and plantain
        peels in a 0.691:1 ratio (BP1u) led to higher biogas production than
        either equal-proportion digestion (BP2u) or single-substrate digestion
        (Bu, Pu). The yield differences among substrates were statistically
        significant.
        """,
        "Results",
    ),
    _paper(
        """
        Slurry concentration significantly impacted biogas production in both
        single and combined digestions. Yields rose with higher slurry
        concentrations, with the 1:6 concentration (C1) producing the most
        biogas, followed by C2 and then C3 with the least. Table 3 (two-way
        ANOVA with replication) shows both slurry concentration and substrate
        type had highly significant effects on biogas yields, with a notable
        interaction between the two factors: Substrate SS = 819.48, df = 3,
        F = 11.93, p = 1.62e-07; Slurry Concentration SS = 964.58, df = 2,
        F = 21.06, p = 1.87e-09; Interaction SS = 1053.58, df = 6, F = 7.67,
        p = 7.68e-08 (all significant at F crit thresholds of 2.63, 3.02 and
        2.12 respectively).
        """,
        "Results",
    ),
    _paper(
        """
        Conclusion. Both slurry concentration and co-digestion significantly
        affected biogas production from the agricultural wastes of bean peels
        and plantain peels. Whether the substrates are digested alone or in
        combination, careful attention must be given to the final slurry
        concentration: higher slurry concentrations generate more biogas than
        dilute slurry concentrations. The study recommends exploring more
        agricultural substrates for their potential to produce biogas, to
        sanitize the environment while augmenting energy needs.
        """,
        "Conclusion",
    ),
]


# --------------------------------------------------------------------------- #
# Domain context -- general AD process knowledge, NOT measured in this study
# --------------------------------------------------------------------------- #

_DOMAIN_DOCS: list[CorpusDocument] = [
    _context(
        """
        Volatile fatty acid (VFA) accumulation is the most common cause of
        anaerobic digester instability. Acidogenic and acetogenic bacteria
        convert hydrolysed organic matter into short-chain fatty acids (acetic,
        propionic, butyric) faster than slower-growing methanogens can consume
        them, especially early in digestion or when readily degradable substrate
        is abundant. If VFA production outpaces the buffering (alkalinity)
        capacity of the slurry, pH falls; below approximately pH 6.5,
        methanogenic activity is measurably inhibited, and below pH 6.0 the
        process is generally considered to have "soured" -- gas production
        collapses and can take days to weeks to recover once the imbalance is
        corrected.
        """,
        "Process chemistry",
        "General anaerobic digestion process literature",
    ),
    _context(
        """
        Alkalinity in an anaerobic digester is primarily supplied by ammonia
        released during protein and amino-acid degradation, which reacts with
        carbon dioxide to form an ammonium bicarbonate buffer. Substrates with
        higher nitrogen content (lower C/N ratio) therefore tend to buffer VFA
        accumulation more effectively, all else equal -- though very low C/N
        ratios (below roughly 20-25:1) can eventually cause ammonia toxicity at
        high loading. A C/N ratio in the range of 20-30:1 is often cited as
        near-optimal for stable co-digestion, though many single agricultural
        substrates, including bean and plantain peel chaff, fall well below this
        and still digest successfully at moderate loading.
        """,
        "Process chemistry",
        "General anaerobic digestion process literature",
    ),
    _context(
        """
        In unseeded (uninoculated) anaerobic digestion, the methanogenic
        consortium must establish itself from whatever microflora is naturally
        present on the substrate, rather than from an already-active seed
        culture such as cow rumen liquor or digested sludge. This typically
        produces a longer and more variable lag phase before measurable biogas
        appears -- often 1-3 weeks rather than the 2-4 days common in seeded
        systems -- and raises the risk that a digester never establishes a
        productive methanogen population within a given observation window,
        particularly for substrates with a nitrogen content too low to buffer
        the initial acid phase.
        """,
        "Process chemistry",
        "General anaerobic digestion process literature",
    ),
    _context(
        """
        Total solids (TS) concentration governs the trade-off between reactor
        loading and mass transfer in anaerobic digestion. Higher %TS increases
        the mass of degradable organic matter per unit volume, which raises
        potential biogas yield, but also increases the concentration of
        inhibitory intermediates (VFA, ammonia) and can impede mixing and
        diffusion, particularly beyond roughly 15-20% TS ("high-solids" or "dry"
        digestion), where mass-transfer limitation and localized acidification
        become more likely. Below this range, yields typically increase
        monotonically with %TS, as more substrate is available without the
        digester approaching mass-transfer or buffering limits.
        """,
        "Process chemistry",
        "General anaerobic digestion process literature",
    ),
    _context(
        """
        The modified Gompertz model is a widely used empirical description of
        cumulative biogas production in batch anaerobic digestion:
        M(t) = P * exp{ -exp[ (Rm * e / P)(lambda - t) + 1 ] }, where M(t) is
        cumulative biogas at time t, P is the ultimate biogas potential, Rm is
        the maximum biogas production rate, lambda is the lag phase duration,
        and e is Euler's number. It captures the characteristic sigmoidal shape
        of batch digestion: little or no gas during the lag phase, a period of
        rapidly increasing rate, and a plateau as substrate is depleted. It has
        been applied in prior work on co-digestion of agricultural peel wastes,
        including plantain peels with cow dung and rice husk with plantain
        peels, to characterise kinetic parameters and compare treatments.
        """,
        "Kinetic modelling",
        "General anaerobic digestion kinetics literature",
    ),
]

CORPUS: list[CorpusDocument] = _PAPER_DOCS + _DOMAIN_DOCS
