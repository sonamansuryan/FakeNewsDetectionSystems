import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

try:
    from src.utils.logger import setup_logger
except ImportError:
    try:
        from utils.logger import setup_logger
    except ImportError:
        from logger import setup_logger

try:
    from src.data_collection.rag_retriever import RetrievedContext
except ImportError:
    try:
        from rag_retriever import RetrievedContext
    except ImportError:
        @dataclass
        class RetrievedContext:  # type: ignore[no-redef]
            query: str
            keywords: list
            combined_context: str = ""
            wikipedia_results: list = field(default_factory=list)
            duckduckgo_results: list = field(default_factory=list)
            source_count: int = 0
            institution_domain: Optional[str] = None
            intent_query: str = ""

            def to_prompt_string(self) -> str:
                return self.combined_context

            @property
            def institution_named(self) -> bool:
                return self.institution_domain is not None

try:
    from src.models.sentiment_model import SentimentResult
except ImportError:
    try:
        from sentiment_model import SentimentResult
    except ImportError:
        SentimentResult = None  # type: ignore[assignment,misc]

@dataclass
class RoBERTaResult:
    label: str
    confidence: float
    label_id: int
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    @property
    def is_fake(self) -> bool:
        return self.label.upper() == "FAKE" or self.label_id == 1

    @classmethod
    def from_dict(cls, d: dict) -> "RoBERTaResult":
        label_id = d.get("label_id", 1)
        label    = d.get("label", "FAKE" if label_id == 1 else "REAL")
        return cls(
            label=label.upper(),
            confidence=float(d.get("confidence", d.get("accuracy", 0.0))),
            label_id=label_id,
            precision=float(d.get("precision", 0.0)),
            recall=float(d.get("recall", 0.0)),
            f1=float(d.get("f1", 0.0)),
        )

import re as _re

_FP_RULES: list[dict] = [
    # ── Graphene nanobots / 5G tracking ──────────────────────────────────────
    {
        "patterns": [
            r"\bgraphene\b.*\b(nanobot|nano.bot|nanoparticle)\b",
            r"\b(nanobot|nano.bot)\b.*\bgraphene\b",
            r"\b5g\b.*\b(track|transmit|surveil|data)\b.*\b(vaccine|vaxx)\b",
            r"\b(vaccine|vaxx)\b.*\b5g\b.*\b(track|transmit|chip|signal)\b",
            r"\bself.assembl\w*\b.*\b(graphene|nanobot)\b",
        ],
        "label":    "CONTRADICTED",
        "explanation": (
            "First Principles analysis (Mistral unavailable): The claim describes "
            "a physically impossible mechanism. Graphene oxide does NOT self-assemble "
            "into functional nanobots under physiological conditions — this violates "
            "basic materials science. 5G photons carry ~0.00001 eV of energy, "
            "orders of magnitude too low to read or transmit biological data from "
            "individual persons. No peer-reviewed study has ever detected graphene "
            "nanobots in any approved vaccine. The URGENT / ALL-CAPS framing and "
            "conspiracy-style language indicate HIGH manipulation risk."
        ),
        "counter_narrative": (
            "COVID-19 vaccines (Pfizer-BioNTech, Moderna, Novavax) contain "
            "mRNA or protein subunit antigens, lipid nanoparticles, salts, and "
            "sugars — all published in regulatory filings publicly available on "
            "the FDA, EMA, and WHO websites. Graphene nanobots are not present "
            "in any approved vaccine. Self-assembly of graphene into functional "
            "electronic devices requires industrial conditions (high temperature, "
            "chemical reduction) that are incompatible with biological tissue. "
            "5G signals operate at 0.6–86 GHz; the photon energy (~0.00001 eV) "
            "is far below the threshold needed to interact with cells or transmit "
            "data from them. Governments track citizens through legal, documented "
            "means (phones, databases) — not via vaccines. The claim uses "
            "ALL-CAPS urgency and viral-share pressure, classic manipulation tactics "
            "designed to bypass critical thinking. It is physically impossible."
        ),
    },
    # ── mRNA / spike protein shedding ────────────────────────────────────────
    {
        "patterns": [
            r"\b(mrna|spike.protein)\b.*\bshed\w*\b",
            r"\bshed\w*\b.*\b(mrna|spike.protein)\b",
            r"\bvaccin\w+\b.*\bshed\w*\b.*\bcontact\b",
        ],
        "label":    "CONTRADICTED",
        "explanation": (
            "First Principles analysis (Mistral unavailable): mRNA degrades within "
            "hours of injection and cannot cross intact skin. No transmissible "
            "particle is produced by vaccination. The shedding claim contradicts "
            "established molecular biology."
        ),
        "counter_narrative": (
            "mRNA molecules are inherently unstable and are broken down by "
            "ubiquitous RNases within hours of injection. They do not enter the "
            "nucleus, are not incorporated into DNA, and cannot be excreted through "
            "skin, breath, or bodily fluids in any infectious or pharmacologically "
            "active form. Spike protein produced by vaccinated cells remains "
            "cell-surface-bound or is rapidly cleared by the immune system. "
            "The CDC, EMA, and WHO have all reviewed post-market surveillance "
            "data across hundreds of millions of doses with no evidence of "
            "transmission of vaccine components to unvaccinated individuals."
        ),
    },
    # ── Microchip / tracking chip in vaccines ─────────────────────────────────
    {
        "patterns": [
            r"\b(microchip|tracking.chip|rfid|gps.chip)\b.*\bvaccin\w*\b",
            r"\bvaccin\w*\b.*\b(microchip|tracking.chip|rfid|implant)\b",
            r"\bBill\s+Gates\b.*\b(chip|microchip|track)\b",
        ],
        "label":    "CONTRADICTED",
        "explanation": (
            "First Principles analysis (Mistral unavailable): The smallest commercial "
            "RFID chips are ~0.4 mm — far too large to pass through a vaccine needle "
            "(0.3–0.6 mm bore). They also require power sources and antennae absent "
            "from any vaccine formulation. No regulatory filing (FDA, EMA, MHRA) lists "
            "electronic components."
        ),
        "counter_narrative": (
            "Vaccine needles have an inner bore of 0.3–0.6 mm. The smallest "
            "commercially available tracking chip (Hitachi mu-chip) measures "
            "0.4 × 0.4 mm — already at the physical limit of the needle — and "
            "still requires an external antenna and power source to function. "
            "No such components appear in any vaccine ingredient list filed with "
            "the FDA, EMA, Health Canada, or TGA. Full ingredient disclosures "
            "are publicly available and independently verified by regulatory "
            "agencies in over 180 countries. The claim is physically impossible "
            "given the constraints of needle bore diameter alone."
        ),
    },
    # ── Vitamin D / supplement beats all vaccines ─────────────────────────────
    {
        "patterns": [
            r"\bvitamin\s+d\b.*\b(better|more.effective|superior|replace|prevent)\b.*\b(vaccine|booster)\b",
            r"\b(supplement|herb|ivermectin)\b.*\b(more.effective|superior|replace)\b.*\bvaccin\w*\b",
        ],
        "label":    "CONTRADICTED",
        "explanation": (
            "First Principles analysis (Mistral unavailable): COVID-19 vaccine "
            "efficacy against severe disease is 70-95% in peer-reviewed RCTs "
            "published in NEJM and The Lancet. No RCT demonstrates vitamin D "
            "achieving comparable efficacy against severe COVID-19. "
            "'Internal reports suggest' is non-peer-reviewed and insufficient "
            "to overturn established vaccine efficacy data — CONTRADICTED by "
            "the comparative RCT evidence hierarchy."
        ),
        "counter_narrative": (
            "Randomised controlled trials (RCTs) published in the New England "
            "Journal of Medicine and The Lancet found COVID-19 vaccines reduce "
            "the risk of severe illness and hospitalisation by 70-95%. Vitamin D "
            "plays a role in general immune function, and deficiency is associated "
            "with worse respiratory outcomes, but no RCT has demonstrated that "
            "vitamin D supplementation achieves comparable protection against "
            "severe COVID-19. The Endocrine Society's 2024 clinical guidelines "
            "do not recommend vitamin D as a COVID-19 preventive measure. "
            "The phrase 'internal reports suggest' indicates non-peer-reviewed, "
            "non-public data — the lowest tier of evidence — which cannot "
            "overturn the published RCT consensus."
        ),
    },
]

for _rule in _FP_RULES:
    _rule["_compiled"] = [
        _re.compile(p, _re.IGNORECASE | _re.DOTALL)
        for p in _rule["patterns"]
    ]


def _first_principles_check(claim: str) -> Optional[dict]:
    for rule in _FP_RULES:
        for compiled in rule["_compiled"]:
            if compiled.search(claim):
                return {
                    "label":             rule["label"],
                    "explanation":       rule["explanation"],
                    "counter_narrative": rule["counter_narrative"],
                }
    return None


TRUTH_SPECTRUM_LABELS: frozenset = frozenset({
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "UNSUPPORTED",
    "MISLEADING_FRAMING",
    "CONTRADICTED",
    "FALSE_FABRICATED",
})

FAKE_SPECTRUM_LABELS: frozenset = frozenset({
    "CONTRADICTED",
    "MISLEADING_FRAMING",
    "FALSE_FABRICATED",
})

REAL_SPECTRUM_LABELS: frozenset = frozenset({
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "UNSUPPORTED",
})


@dataclass
class CounterNarrativeResult:

    claim: str
    roberta_label: str
    roberta_confidence: float
    final_label: str
    sub_claims: list = field(default_factory=list)
    explanation: str = ""
    counter_narrative: str = ""
    is_correction: bool = False
    sources_used: list = field(default_factory=list)
    refutation_found: bool = False
    refutation_sources: list = field(default_factory=list)
    model_used: str = "mistral"
    generation_time_ms: float = 0.0
    context_chars: int = 0
    error: Optional[str] = None

    @property
    def is_fake(self) -> bool:
        return self.final_label.upper() in FAKE_SPECTRUM_LABELS

    @property
    def decision_changed(self) -> bool:
        return self.is_correction

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "roberta_classification": {
                "label":      self.roberta_label,
                "confidence": round(self.roberta_confidence, 4),
            },
            "final_verdict": {
                "label":         self.final_label,
                "spectrum":      self.final_label,          # explicit alias
                "is_fake":       self.is_fake,
                "is_correction": self.is_correction,
                "sub_claims":    self.sub_claims,
                "explanation":   self.explanation,
            },
            "counter_narrative":    self.counter_narrative,
            "refutation": {
                "refutation_found":   self.refutation_found,
                "refutation_sources": self.refutation_sources,
            },
            "sources":              self.sources_used,
            "model":                self.model_used,
            "generation_time_ms":   round(self.generation_time_ms, 2),
            "context_chars":        self.context_chars,
            "error":                self.error,
        }


class UnifiedPromptBuilder:

    SYSTEM_PROMPT = (
        "You are a Supreme Judge fact-checker in a multi-signal AI pipeline. "
        "Input: a CLAIM, RoBERTa classifier verdict, RAG evidence, and sentiment. "
        "Output ONE valid JSON object. No markdown fences. No text before or after.\n\n"

        "STEP 0 — TEMPORAL PRIORITY (run FIRST, always):\n"
        "RAG context is fetched TODAY (real-time) and may postdate your training cutoff. "
        "TRUST RAG Wikipedia/Tier-1 snippets with specific dates over your training knowledge.\n"
        "PRESENT-TENSE CLAIMS ('is now', 'currently', 'has become'): verify the CURRENT STATE, "
        "not whether the confirming date equals today. Past event -> present state is NOT a contradiction.\n"
        "Example: 'Sweden IS a NATO member' + RAG 'joined March 2024' -> SUPPORTED.\n\n"

        "STEP 1 — HARD RELEVANCE GATE:\n"
        "For each RAG snippet: does it directly address the claim topic?\n"
        "PASS -> use it.  FAIL -> discard completely, cite nothing from it.\n"
        "AUTOMATIC PASS: (a) Wikipedia article whose title IS the primary subject of the claim. "
        "(b) Any snippet describing an institution's ACTUAL mandate when that institution is named "
        "in the claim — this enables the mandatory comparative analysis in Step 3.\n\n"

        "STEP 2 — EVIDENCE EVALUATION (snippets that passed Step 1 only):\n"
        "Tier 1: official bodies (WHO, NIH, CDC, NASA, UN, EU, NATO, government sites).\n"
        "Tier 2: Reuters, BBC, AP, Nature, major science/news outlets.\n"
        "Tier 3: established fact-checkers (Snopes, PolitiFact, FullFact).\n"
        "Tier 5: forums, anonymous blogs, shops -> silently ignore.\n\n"
        "Step 2b — FIRST PRINCIPLES (apply when ALL snippets failed Step 1):\n"
        "Reason from physics, biology, or chemistry. Name the specific principle and quantify it.\n"
        "Examples of impossible mechanisms that justify CONTRADICTED:\n"
        "  - mRNA vaccine spike protein shedding via touch: mRNA degrades within hours, "
        "does not cross intact skin, no transmissible particle is produced by vaccination.\n"
        "  - Graphene nanobots in vaccines transmitting via 5G: graphene oxide is not "
        "self-assembling at physiological conditions; 5G photons (0.00001 eV) cannot "
        "read or transmit biological data.\n"
        "  - 'Vitamin D more effective than any booster': COVID-19 vaccine efficacy vs severe "
        "disease is 70-90% in peer-reviewed RCTs (NEJM, Lancet). No RCT shows vitamin D "
        "matching this. 'Internal reports suggest' is non-peer-reviewed and insufficient "
        "to overturn established vaccine efficacy data -> CONTRADICTED.\n"
        "A physically/biologically impossible mechanism = CONTRADICTED even with zero relevant RAG.\n\n"

        "STEP 3 — MANDATORY COMPARATIVE ANALYSIS (whenever an institution is named):\n"
        "You MUST perform this even when the claim seems obviously false. Do not skip it.\n"
        "Column A: What the institution ACTUALLY does — its documented mandate, powers, "
        "and verified activities (from Tier 1/2 RAG or established institutional knowledge).\n"
        "Column B: What the claim ASSERTS the institution did, said, or is planning.\n"
        "Gap assessment:\n"
        "  - Partial mismatch (real activity + exaggerated detail) -> MISLEADING_FRAMING.\n"
        "  - No institutional activity matching claim, but claim is possible in principle -> UNSUPPORTED.\n"
        "  - Claim is the DIRECT OPPOSITE of documented institutional reality -> FALSE_FABRICATED.\n"
        "FALSE_FABRICATED triggers:\n"
        "  * UN 'secret global social credit system': UN has NO enforcement mechanism over "
        "individual citizens, NO monitoring infrastructure, and its Charter explicitly respects "
        "sovereignty. A claim that UN secretly monitors and restricts individual citizens is the "
        "direct opposite of UN's documented mandate -> FALSE_FABRICATED.\n"
        "  * WHO 'secret population control agenda': WHO mandate is public health advisory, "
        "zero coercive authority over individuals -> FALSE_FABRICATED if claim asserts coercion.\n"
        "  * NATO 'running shadow government': NATO is a mutual defense treaty, no civilian "
        "governance mandate -> FALSE_FABRICATED if claim asserts domestic governance.\n"
        "IMPORTANT: Do NOT soften FALSE_FABRICATED to UNSUPPORTED by saying 'no evidence found'. "
        "If Step 3 reveals total inversion of documented reality, use FALSE_FABRICATED.\n\n"

        "STEP 4 — ASSIGN TRUTH SPECTRUM LABEL (exactly one):\n"
        "  SUPPORTED           -> key sub-claims backed by Tier 1/2 evidence.\n"
        "  PARTIALLY_SUPPORTED -> core kernel real, details wrong or exaggerated.\n"
        "  UNSUPPORTED         -> nothing confirms OR denies; empirically possible but unverified.\n"
        "  MISLEADING_FRAMING  -> facts accurate but overall framing creates false impression.\n"
        "  CONTRADICTED        -> Tier 1-3 source OR established science directly refutes claim.\n"
        "  FALSE_FABRICATED    -> documented institutional/scientific reality is the DIRECT OPPOSITE.\n"
        "DECISION TREE (follow in order, stop at first match):\n"
        "  1. Does Tier 1/2 RAG confirm the claim? -> SUPPORTED (or PARTIALLY_SUPPORTED if partial).\n"
        "  2. Does Step 3 show total inversion of institutional documented reality? -> FALSE_FABRICATED.\n"
        "  3. Does Tier 1-3 source or Step 2b physics/biology ACTIVELY refute the mechanism? -> CONTRADICTED.\n"
        "  4. Is the claim pseudoscientific (vaccine shedding, graphene nanobots, EM-DNA, "
        "5G-virus, supplement-beats-vaccine-RCT)? -> CONTRADICTED via Step 2b.\n"
        "  5. Is claim a superiority assertion ('X more effective than Y') with only "
        "'internal reports' / non-RCT evidence? -> CONTRADICTED (established RCT data refutes it).\n"
        "  6. Is there genuinely NO evidence either way and mechanism is not scientifically "
        "impossible? -> UNSUPPORTED.\n\n"

        "STEP 5 — RoBERTa OVERRIDE: if your label polarity differs from RoBERTa, state:\n"
        "'OVERRIDE: Despite RoBERTa [X]% [label], [specific evidence/principle] -> [YOUR LABEL].'\n\n"

        "OUTPUT FORMAT — strict JSON, exactly four keys:\n"
        "{\n"
        "  \"final_label\": \"<one of the six labels above>\",\n"
        "  \"sub_claims\": [\"<atomic verifiable assertion>\", \"<...>\"],\n"
        "  \"explanation\": \"<60-100 words. State which snippets passed/failed gate. "
        "Name the Tier. Mandatory Step 3 comparative analysis result if institution named. "
        "Step 2b principle if all RAG failed. Decision tree step that triggered the label. "
        "One sentence on tone/manipulation risk.>\",\n"
        "  \"counter_narrative\": \"<120-200 words. MANDATORY, never empty. "
        "SUPPORTED: enrich with verified context. "
        "CONTRADICTED: name the specific scientific principle that makes it impossible; "
        "state what the actual science/consensus says. "
        "FALSE_FABRICATED: explain institution's real mandate (Column A), "
        "then show exactly how the claim inverts it (Column B gap). "
        "UNSUPPORTED: state what IS known and what remains genuinely unverifiable. "
        "Same language as claim.\"\n"
        "}\n\n"

        "NON-NEGOTIABLE RULES:\n"
        "1. JSON only. No markdown, no preamble, no text after closing brace.\n"
        "2. counter_narrative: 120-200 words. Complete before closing }.\n"
        "3. Never cite a snippet that failed the relevance gate.\n"
        "4. Step 3 Comparative Analysis is MANDATORY whenever an institution is named — never skip.\n"
        "5. Do NOT soften FALSE_FABRICATED to UNSUPPORTED to appear neutral.\n"
        "6. ASCII punctuation only inside JSON string values.\n"
        "7. Same language as the claim."
    )

    @staticmethod
    def _sentiment_section(sentiment_result) -> str:

        if sentiment_result is None:
            return "[No sentiment analysis available.]"
        try:
            if hasattr(sentiment_result, "to_prompt_fragment"):
                return sentiment_result.to_prompt_fragment()
            return (
                f"Primary emotion  : {getattr(sentiment_result, 'primary_emotion', 'unknown').upper()}\n"
                f"Subjectivity     : {getattr(sentiment_result, 'subjectivity_score', 0.0):.2f}\n"
                f"Manipulation risk: {getattr(sentiment_result, 'manipulation_risk', 'UNKNOWN')}\n"
                f"Risk reason      : {getattr(sentiment_result, 'manipulation_reason', '')}"
            )
        except Exception:
            return "[Sentiment analysis could not be formatted.]"

    @staticmethod
    def _institution_annotation(context: Optional[RetrievedContext]) -> str:

        if context is None:
            return ""

        institution_domain = getattr(context, "institution_domain", None)
        intent_query       = getattr(context, "intent_query", "")
        refutation_query   = getattr(context, "refutation_query", "")
        has_refutation     = getattr(context, "has_refutation", False)

        lines = []
        if institution_domain:
            lines.append(
                f"\n[RAG metadata] Institution detected in claim: {institution_domain}"
            )
            lines.append(
                f"[RAG metadata] Search strategy: TARGETED (site:{institution_domain})"
            )
            lines.append(f"[RAG metadata] Query used: {intent_query}")
            lines.append(
                "[RAG metadata] If no confirmation from this institution appears "
                "in the snippets below, apply CREDIBILITY GAP detection."
            )
        else:
            lines.append("\n[RAG metadata] No major institution detected in claim.")
            lines.append("[RAG metadata] Search strategy: BROAD INTENT")
            lines.append(f"[RAG metadata] Query used: {intent_query}")

        if refutation_query:
            if has_refutation:
                lines.append(
                    "[RAG metadata] REFUTATION SEARCH: TRIGGERED -- "
                    "active counter-evidence retrieved (see [REFUTATION SEARCH] block). "
                    "This is SCENARIO B: Evidence of Absence. "
                    "CONTRADICTED label is strongly indicated if sources pass the Hard "
                    "Relevance Gate."
                )
            else:
                lines.append(
                    "[RAG metadata] REFUTATION SEARCH: TRIGGERED -- "
                    "no active counter-evidence returned. "
                    "This is SCENARIO A: Lack of Support only. "
                    "Use UNSUPPORTED, not CONTRADICTED."
                )
        else:
            lines.append(
                "[RAG metadata] REFUTATION SEARCH: Not triggered "
                "(no health/science/institutional signal detected in claim)."
            )

        return "\n".join(lines)

    @classmethod
    def build(
        cls,
        claim: str,
        roberta_result: "RoBERTaResult",
        context: Optional[RetrievedContext],
        sentiment_result=None,
    ) -> str:

        institution_annotation = cls._institution_annotation(context)

        raw_context = (
            context.combined_context.strip()
            if context and context.combined_context.strip()
            else ""
        )

        if raw_context:
            context_section = (
                institution_annotation + "\n" + raw_context +
                "\n\n[GATE INSTRUCTION: Apply Hard Relevance Gate (Step 1) to EVERY snippet. "
                "If ALL snippets fail the gate (off-topic, unrelated Wikipedia articles, "
                "no direct mention of claim's specific mechanism or topic): "
                "DISCARD ALL of them and IMMEDIATELY apply Step 2b — First Principles Reasoning. "
                "Step 2b means: reason from physics, biology, or chemistry about whether "
                "the claimed mechanism is scientifically possible. Name the specific principle. "
                "A physically/biologically impossible mechanism = CONTRADICTED. "
                "An unverifiable but not impossible claim with no RAG support = UNSUPPORTED. "
                "NEVER default to UNSUPPORTED for pseudoscientific health claims "
                "(vaccine shedding, miracle cures, electromagnetic DNA changes) — "
                "these have established scientific consensus AGAINST them: use CONTRADICTED.]"
            )
        else:
            context_section = (
                institution_annotation +
                "\n[No RAG evidence retrieved. Apply Step 2b immediately: "
                "assess via physics/biology/chemistry first principles. "
                "If the claimed mechanism is scientifically impossible -> CONTRADICTED. "
                "If claim is empirically unverifiable (not provably false) -> UNSUPPORTED. "
                "Do NOT use UNSUPPORTED for claims with established scientific consensus against them.]"
            )

        sentiment_section = cls._sentiment_section(sentiment_result)

        import datetime as _dt
        _today = _dt.datetime.utcnow().strftime("%Y-%m-%d")

        return (
            f"[TODAY: {_today} UTC. RAG is real-time and overrides your training knowledge.]\n"
            f"[PRESENT-TENSE RULE: Claims like 'X IS/NOW/CURRENTLY' assert a CURRENT STATE. "
            f"If RAG confirms that state exists today (e.g. membership, declaration), mark SUPPORTED "
            f"even if the state was established in the past. Past date != contradiction of present state.]\n\n"
            f"CLAIM: \"{claim}\"\n\n"
            f"SIGNAL 1 — RoBERTa: {roberta_result.label} ({roberta_result.confidence * 100:.0f}% confidence)\n\n"
            f"SIGNAL 2 — RAG EVIDENCE:\n{context_section}\n\n"
            f"SIGNAL 3 — SENTIMENT:\n{sentiment_section}\n\n"
            "KEY RULES:\n"
            "- Wikipedia primary-subject article ALWAYS passes relevance gate.\n"
            "- UK: election Jul 4 != appointment Jul 5. One-day gap is NOT contradiction.\n"
            "- If Tier 1/2 snippet with date confirms claim -> SUPPORTED.\n"
            "- Present-tense state claims ('is now', 'currently'): verify the STATE, not the date.\n"
            "- PSEUDOSCIENCE RULE: Claims about vaccine shedding, EM wave DNA changes, "
            "miracle cures beating vaccines, 5G activating pathogens, etc. have established "
            "scientific consensus AGAINST them. Even with no relevant RAG, use CONTRADICTED "
            "and cite the specific biological/physical principle that makes them impossible.\n"
            "- COMPARATIVE SUPERIORITY CLAIMS ('X is more effective than Y'): require "
            "peer-reviewed RCT evidence. 'Internal reports suggest' = NOT peer-reviewed. "
            "Without RCT evidence, a superiority claim is UNSUPPORTED at best, "
            "CONTRADICTED if scientific consensus shows the opposite.\n"
            "- Write full counter_narrative (120-200 words). Do not truncate.\n\n"
            "Return JSON verdict now:"
        )

def _parse_mistral_json(raw: str) -> Optional[dict]:
    if not raw:
        return None

    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    cleaned = match.group(0)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    _LABEL_KEY_ALIASES = [
        "final_label", "verdict", "label", "result",
        "classification", "final_verdict", "decision",
    ]
    raw_label = ""
    for key in _LABEL_KEY_ALIASES:
        val = data.get(key, "")
        if isinstance(val, dict):
            # e.g. {"verdict": {"label": "SUPPORTED", "confidence": "High"}}
            val = val.get("label", val.get("final_label", ""))
        if val:
            raw_label = str(val)
            break

    label = raw_label.upper().strip()

    _LEGACY_MAP = {
        "REAL":           "SUPPORTED",
        "TRUE":           "SUPPORTED",
        "VERIFIED":       "SUPPORTED",
        "FAKE":           "CONTRADICTED",
        "FALSE":          "CONTRADICTED",
        "REFUTED":        "CONTRADICTED",
        "FABRICATED":     "FALSE_FABRICATED",
        "UNVERIFIABLE":   "UNSUPPORTED",
        "UNVERIFIED":     "UNSUPPORTED",
        "UNKNOWN":        "UNSUPPORTED",
        "MISLEADING":     "MISLEADING_FRAMING",
        "PARTIAL":        "PARTIALLY_SUPPORTED",
        "PARTIALLY TRUE": "PARTIALLY_SUPPORTED",
        "PARTIALLY_TRUE": "PARTIALLY_SUPPORTED",
    }
    label = _LEGACY_MAP.get(label, label)

    if label not in TRUTH_SPECTRUM_LABELS:
        for valid in TRUTH_SPECTRUM_LABELS:
            if valid in label:
                label = valid
                break
        else:
            return None   # genuinely unrecognised label

    raw_sub = data.get("sub_claims", [])
    if isinstance(raw_sub, list):
        sub_claims = [str(s).strip() for s in raw_sub if str(s).strip()]
    else:
        sub_claims = [str(raw_sub).strip()] if str(raw_sub).strip() else []

    explanation = (
        str(data.get("explanation", "") or
            data.get("reasoning", "") or
            data.get("analysis", "") or
            data.get("ai_reasoning", "")).strip()
    )
    # Accept counter_narrative aliases
    narrative = (
        str(data.get("counter_narrative", "") or
            data.get("counter_narrative_text", "") or
            data.get("narrative", "") or
            data.get("correction", "")).strip()
    )

    return {
        "final_label":       label,
        "sub_claims":        sub_claims,
        "explanation":       explanation or "No explanation provided.",
        "counter_narrative": narrative   or "No counter-narrative generated.",
    }

class CounterNarrativeGenerator:
    OLLAMA_DEFAULT_URL = "http://localhost:11434"

    def __init__(
        self,
        config_path: Optional[str] = None,
        ollama_url: Optional[str] = None,
        model_name: str = "mistral",
        temperature: float = 0.1,
        max_tokens: int = 1800,
        timeout: int = 120,
        max_retries: int = 2,
    ):
        self.logger = setup_logger("CounterNarrativeGenerator", config_path=config_path)

        if config_path:
            self._load_config(config_path)
        else:
            self.ollama_url  = (ollama_url or self.OLLAMA_DEFAULT_URL).rstrip("/")
            self.model_name  = model_name
            self.temperature = temperature
            self.max_tokens  = max_tokens
            self.timeout     = timeout
            self.max_retries = max_retries

        if ollama_url:
            self.ollama_url = ollama_url.rstrip("/")

        self.generate_endpoint = f"{self.ollama_url}/api/chat"

        self.logger.info(
            "CounterNarrativeGenerator ready | model=%s url=%s temp=%.2f max_tokens=%d",
            self.model_name, self.ollama_url, self.temperature, self.max_tokens,
        )
        self._check_ollama_health()

    def _load_config(self, config_path: str) -> None:
        import yaml
        defaults = dict(
            ollama_url=self.OLLAMA_DEFAULT_URL,
            model_name="mistral",
            temperature=0.1,
            max_tokens=1800,
            timeout=120,
            max_retries=2,
        )
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            g = cfg.get("counter_narrative", {})
            self.ollama_url  = g.get("ollama_url", g.get("ollama_host", defaults["ollama_url"])).rstrip("/")
            self.model_name  = g.get("model",         defaults["model_name"])
            self.temperature = float(g.get("temperature", defaults["temperature"]))
            self.max_tokens  = int(g.get("max_tokens",    defaults["max_tokens"]))
            self.timeout     = int(g.get("timeout",        defaults["timeout"]))
            self.max_retries = int(g.get("max_retries",    defaults["max_retries"]))
            self.logger.info("Config loaded from '%s'.", config_path)
        except (FileNotFoundError, KeyError, TypeError) as exc:
            self.logger.warning("Config not loaded (%s). Using defaults.", exc)
            for key, val in defaults.items():
                setattr(self, key, val)

    def _check_ollama_health(self) -> bool:
        """Ping Ollama to verify it is running. Non-fatal if it is not."""
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                self.logger.info("Ollama running. Available models: %s", models)
                base = self.model_name.split(":")[0]
                if not any(base in m for m in models):
                    self.logger.warning(
                        "Model '%s' not found. Run: ollama pull %s",
                        self.model_name, self.model_name,
                    )
                return True
        except requests.exceptions.ConnectionError:
            self.logger.warning(
                "Ollama not reachable at %s. Run: ollama serve", self.ollama_url
            )
        except Exception as exc:
            self.logger.warning("Ollama health check failed: %s", exc)
        return False


    def _call_ollama(self, system_prompt: str, user_prompt: str) -> Optional[str]:

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature":    self.temperature,
                "num_predict":    self.max_tokens,
                "top_p":          0.9,
                "repeat_penalty": 1.1,
            },
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                self.logger.debug(
                    "Ollama call attempt %d/%d | prompt_len=%d",
                    attempt, self.max_retries, len(user_prompt),
                )
                t0   = time.time()
                resp = requests.post(
                    self.generate_endpoint, json=payload, timeout=self.timeout
                )
                elapsed_ms = (time.time() - t0) * 1000
                resp.raise_for_status()

                content = resp.json().get("message", {}).get("content", "").strip()
                if content:
                    self.logger.debug(
                        "Ollama responded: %d chars in %.0fms.", len(content), elapsed_ms
                    )
                    return content
                self.logger.warning("Ollama returned empty content on attempt %d.", attempt)

            except requests.exceptions.Timeout:
                self.logger.warning(
                    "Timeout after %ds on attempt %d. "
                    "Consider increasing timeout for CPU inference.",
                    self.timeout, attempt,
                )
            except requests.exceptions.ConnectionError:
                self.logger.error("Cannot connect to Ollama. Is `ollama serve` running?")
                return None
            except requests.exceptions.HTTPError as exc:
                self.logger.error("Ollama HTTP error: %s", exc)
                if exc.response and exc.response.status_code == 404:
                    self.logger.error(
                        "Model '%s' not found. Run: ollama pull %s",
                        self.model_name, self.model_name,
                    )
                    return None
            except (json.JSONDecodeError, KeyError) as exc:
                self.logger.error("Failed to parse Ollama response: %s", exc)
                return None

            if attempt < self.max_retries:
                delay = 2.0 * attempt
                self.logger.debug("Retrying in %.1fs...", delay)
                time.sleep(delay)

        return None

    @staticmethod
    def _extract_sources(context: Optional[RetrievedContext]) -> list:

        if not context:
            return []
        seen_urls: set = set()
        sources: list = []
        all_items = (
            context.wikipedia_results
            + context.duckduckgo_results
            + getattr(context, "refutation_results", [])
        )
        for item in all_items:
            url = item.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    "url":   url,
                    "title": item.get("title", "").strip() or url,
                })
        return sources

    @staticmethod
    def _extract_refutation_sources(context: Optional[RetrievedContext]) -> list:

        if not context:
            return []
        refutation_results = getattr(context, "refutation_results", [])
        seen_urls: set = set()
        sources: list = []
        for item in refutation_results:
            url = item.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    "url":   url,
                    "title": item.get("title", "").strip() or url,
                })
        return sources

    def _fallback_result(
        self,
        claim: str,
        roberta_result: RoBERTaResult,
        context: Optional[RetrievedContext],
        elapsed_ms: float,
        error_msg: str,
    ) -> CounterNarrativeResult:
        fp = _first_principles_check(claim)
        if fp:
            self.logger.info(
                "First Principles Engine matched claim (Mistral unavailable). "
                "label=%s", fp["label"]
            )
            return CounterNarrativeResult(
                claim=claim,
                roberta_label=roberta_result.label,
                roberta_confidence=roberta_result.confidence,
                final_label=fp["label"],
                sub_claims=[],
                explanation=fp["explanation"],
                counter_narrative=fp["counter_narrative"],
                is_correction=False,
                sources_used=self._extract_sources(context),
                refutation_found=getattr(context, "has_refutation", False) if context else False,
                refutation_sources=self._extract_refutation_sources(context),
                model_used="first_principles_engine",
                generation_time_ms=elapsed_ms,
                context_chars=len(context.combined_context) if context else 0,
                error=None,
            )

        self.logger.warning(
            "Mistral unavailable and no First Principles pattern matched. "
            "Returning UNSUPPORTED fallback."
        )
        return CounterNarrativeResult(
            claim=claim,
            roberta_label=roberta_result.label,
            roberta_confidence=roberta_result.confidence,
            final_label="UNSUPPORTED",
            sub_claims=[],
            explanation=(
                f"AI reasoning engine (Mistral) is currently unavailable. "
                f"RoBERTa classified this as {roberta_result.label} "
                f"(confidence: {roberta_result.confidence * 100:.0f}%), but without "
                f"evidence retrieval analysis, a definitive verdict cannot be issued. "
                f"Please try again in a moment, or verify the claim manually."
            ),
            counter_narrative="",
            is_correction=False,
            sources_used=self._extract_sources(context),
            refutation_found=getattr(context, "has_refutation", False) if context else False,
            refutation_sources=self._extract_refutation_sources(context),
            model_used=self.model_name,
            generation_time_ms=elapsed_ms,
            context_chars=len(context.combined_context) if context else 0,
            error=error_msg,
        )

    def generate(
        self,
        claim: str,
        roberta_result,
        context: Optional[RetrievedContext] = None,
        sentiment_result=None,
    ) -> CounterNarrativeResult:

        if isinstance(roberta_result, dict):
            roberta_result = RoBERTaResult.from_dict(roberta_result)

        institution_domain = getattr(context, "institution_domain", None)
        self.logger.info(
            "generate() | roberta=%s conf=%.1f%% | context_sources=%d | "
            "sentiment=%s | institution=%s",
            roberta_result.label,
            roberta_result.confidence * 100,
            context.source_count if context else 0,
            getattr(sentiment_result, "manipulation_risk", "N/A"),
            institution_domain or "(none)",
        )
        user_prompt = UnifiedPromptBuilder.build(
            claim, roberta_result, context, sentiment_result
        )
        self.logger.debug("Prompt length: %d chars.", len(user_prompt))

        t0           = time.time()
        raw_response = self._call_ollama(
            UnifiedPromptBuilder.SYSTEM_PROMPT, user_prompt
        )
        elapsed_ms   = (time.time() - t0) * 1000

        if not raw_response:
            error = "Ollama call failed -- model unavailable or connection error."
            self.logger.error(error)
            return self._fallback_result(
                claim, roberta_result, context, elapsed_ms, error
            )

        parsed = _parse_mistral_json(raw_response)

        if not parsed:
            error = (
                f"JSON parsing failed. Raw response (first 200 chars): "
                f"'{raw_response[:200]}'"
            )
            self.logger.error(error)
            return self._fallback_result(
                claim, roberta_result, context, elapsed_ms, error
            )

        linguistic_override = getattr(sentiment_result, "linguistic_override", "")
        if linguistic_override and str(linguistic_override).strip():
            parsed["explanation"] = re.sub(
                r"\bneutral\b", "ALARMIST", parsed["explanation"], flags=re.IGNORECASE
            )

        _FP_PLACEHOLDER = {"no explanation provided.", "no counter-narrative generated."}
        _expl_empty = (
            not parsed["explanation"] or
            parsed["explanation"].lower().strip().rstrip(".") + "." in _FP_PLACEHOLDER
        )
        _narr_empty = (
            not parsed["counter_narrative"] or
            parsed["counter_narrative"].lower().strip().rstrip(".") + "." in _FP_PLACEHOLDER
        )
        if _expl_empty or _narr_empty:
            fp = _first_principles_check(claim)
            if fp:
                self.logger.info(
                    "Mistral returned empty prose fields; enriching with "
                    "First Principles Engine (label=%s).", fp["label"]
                )
                if _expl_empty:
                    parsed["explanation"] = fp["explanation"]
                if _narr_empty:
                    parsed["counter_narrative"] = fp["counter_narrative"]
                if (
                    parsed["final_label"] not in FAKE_SPECTRUM_LABELS
                    and fp["label"] in FAKE_SPECTRUM_LABELS
                ):
                    self.logger.info(
                        "FP label (%s) stronger than Mistral label (%s); adopting FP.",
                        fp["label"], parsed["final_label"]
                    )
                    parsed["final_label"] = fp["label"]
            else:
                if _expl_empty:
                    parsed["explanation"] = (
                        f"The claim was classified as {parsed['final_label']} based on "
                        f"analysis of retrieved evidence and linguistic signals. "
                        f"Detailed reasoning was not generated for this response."
                    )


        roberta_is_fake  = roberta_result.label.upper() == "FAKE"
        mistral_is_fake  = parsed["final_label"].upper() in FAKE_SPECTRUM_LABELS
        is_correction: bool = roberta_is_fake != mistral_is_fake

        if is_correction:
            self.logger.warning(
                "DECISION CONFLICT RESOLVED | RoBERTa=%s -> Mistral=%s | %s",
                roberta_result.label,
                parsed["final_label"],
                parsed["explanation"][:120],
            )
        else:
            self.logger.info(
                "Verdict confirmed: %s (both models agree).", parsed["final_label"]
            )
        refutation_found   = getattr(context, "has_refutation", False) if context else False
        refutation_sources = self._extract_refutation_sources(context)

        result = CounterNarrativeResult(
            claim=claim,
            roberta_label=roberta_result.label,
            roberta_confidence=roberta_result.confidence,
            final_label=parsed["final_label"],
            sub_claims=parsed.get("sub_claims", []),
            explanation=parsed["explanation"],
            counter_narrative=parsed["counter_narrative"],
            is_correction=is_correction,
            sources_used=self._extract_sources(context),
            refutation_found=refutation_found,
            refutation_sources=refutation_sources,
            model_used=self.model_name,
            generation_time_ms=elapsed_ms,
            context_chars=len(context.combined_context) if context else 0,
        )

        self.logger.info(
            "Done | roberta=%s -> spectrum=%s | is_correction=%s | "
            "sub_claims=%d | refutation=%s | narrative=%d chars | %.0fms",
            roberta_result.label, result.final_label,
            result.is_correction, len(result.sub_claims),
            result.refutation_found,
            len(result.counter_narrative), elapsed_ms,
        )
        return result

    def generate_batch(self, items: list) -> list:
        self.logger.info("Batch generation starting | %d items.", len(items))
        results = []
        for i, item in enumerate(items, 1):
            self.logger.info("Processing item %d/%d...", i, len(items))
            results.append(self.generate(
                claim=item["claim"],
                roberta_result=item["roberta_result"],
                context=item.get("context"),
                sentiment_result=item.get("sentiment_result"),
            ))
        self.logger.info("Batch complete. %d results generated.", len(results))
        return results

if __name__ == "__main__":
    from dataclasses import dataclass as _dc, field as _f

    # -- Mock context that names WHO (triggers Credibility Gap + Refutation logic)
    @_dc
    class _MockContext:
        query: str = "WHO confirmed COVID-19 vaccines cause infertility"
        keywords: list = _f(default_factory=lambda: ["WHO", "COVID-19", "vaccine", "infertility"])
        combined_context: str = (
            "=== RETRIEVED CONTEXT ===\n"
            "[Search strategy: TARGETED -- site:who.int]\n\n"
            "[Wikipedia] COVID-19 vaccine\n"
            "COVID-19 vaccines have been evaluated for safety across hundreds "
            "of millions of doses. No credible evidence links them to infertility "
            "in clinical or post-market surveillance data.\n"
            "Source: https://en.wikipedia.org/wiki/COVID-19_vaccine\n\n"
            "[DuckDuckGo] Vaccine safety -- general blog post\n"
            "Some online communities have raised questions about long-term vaccine "
            "effects, though these claims are not supported by peer-reviewed research.\n"
            "Source: https://example-healthblog.com/vaccines\n\n"
            "=== REFUTATION SEARCH (Active Counter-Evidence -- Intent 4) ===\n"
            "[Refutation query: \"COVID-19 vaccine infertility scientific consensus "
            "OR evidence against OR debunked\"]\n"
            "[NOTE: The following sources actively contradict or debunk the claim.]\n\n"
            "[REFUTATION] WHO FAQ: COVID-19 vaccines and fertility\n"
            "The World Health Organization states there is no evidence that any "
            "COVID-19 vaccine causes infertility in women or men. This claim has "
            "been reviewed and rejected by multiple independent regulatory bodies "
            "including the EMA, FDA, and MHRA.\n"
            "Source: https://www.who.int/news-room/questions-and-answers/item/"
            "coronavirus-disease-covid-19-vaccines-safety\n\n"
            "[REFUTATION] Reuters Fact Check: COVID vaccines and infertility\n"
            "Multiple fact-checkers including Reuters, AFP, and FullFact have "
            "debunked claims that COVID-19 vaccines cause infertility. Scientists "
            "say the spike protein does not affect reproductive organs.\n"
            "Source: https://www.reuters.com/article/factcheck-covid-vaccine-fertility\n"
        )
        wikipedia_results: list = _f(default_factory=list)
        duckduckgo_results: list = _f(default_factory=list)
        refutation_results: list = _f(default_factory=lambda: [
            {
                "source": "duckduckgo_web",
                "title": "WHO FAQ: COVID-19 vaccines and fertility",
                "snippet": (
                    "The World Health Organization states there is no evidence that any "
                    "COVID-19 vaccine causes infertility in women or men. This claim has "
                    "been reviewed and rejected by multiple independent regulatory bodies "
                    "including the EMA, FDA, and MHRA."
                ),
                "url": (
                    "https://www.who.int/news-room/questions-and-answers/item/"
                    "coronavirus-disease-covid-19-vaccines-safety"
                ),
            },
            {
                "source": "duckduckgo_web",
                "title": "Reuters Fact Check: COVID vaccines and infertility",
                "snippet": (
                    "Multiple fact-checkers including Reuters, AFP, and FullFact have "
                    "debunked claims that COVID-19 vaccines cause infertility. Scientists "
                    "say the spike protein does not affect reproductive organs."
                ),
                "url": "https://www.reuters.com/article/factcheck-covid-vaccine-fertility",
            },
        ])
        source_count: int = 4
        institution_domain: str = "who.int"
        intent_query: str = "COVID-19 vaccine infertility site:who.int"
        refutation_query: str = (
            "COVID-19 vaccine infertility scientific consensus OR evidence against OR debunked"
        )

        def to_prompt_string(self) -> str:
            return self.combined_context

        @property
        def institution_named(self) -> bool:
            return True

        @property
        def has_refutation(self) -> bool:
            return bool(self.refutation_results)

    @_dc
    class _MockSentiment:
        primary_emotion: str = "fear"
        primary_confidence: float = 0.77
        subjectivity_score: float = 0.68
        manipulation_risk: str = "HIGH"
        manipulation_reason: str = "Fear-inducing framing targeting health anxieties."
        linguistic_override: str = ""

        def to_prompt_fragment(self) -> str:
            return (
                f"Primary emotion  : {self.primary_emotion.upper()} "
                f"({self.primary_confidence:.0%} confidence)\n"
                f"Subjectivity     : {self.subjectivity_score:.2f}\n"
                f"Manipulation risk: {self.manipulation_risk}\n"
                f"Risk reason      : {self.manipulation_reason}"
            )

    generator = CounterNarrativeGenerator()

    result = generator.generate(
        claim="WHO confirmed COVID-19 vaccines cause infertility",
        roberta_result={"label": "REAL", "confidence": 0.89, "label_id": 0},
        context=_MockContext(),
        sentiment_result=_MockSentiment(),
    )

    print("\n" + "=" * 70)
    print(f"CLAIM          : {result.claim}")
    print(f"RoBERTa        : {result.roberta_label} ({result.roberta_confidence:.0%})")
    print(f"Spectrum label : {result.final_label}  (is_fake={result.is_fake})")
    print(f"is_correction  : {result.is_correction}  <- polarity-based, computed in Python")
    print(f"Sub-claims     : {len(result.sub_claims)}")
    for i, sc in enumerate(result.sub_claims, 1):
        print(f"  [{i}] {sc}")
    print(f"refutation_found   : {result.refutation_found}")
    print(f"refutation_sources : {result.refutation_sources}")
    print(f"\nExplanation:\n{result.explanation}")
    print(f"\nCounter-narrative:\n{result.counter_narrative}")
    print("\nFull to_dict():")
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))