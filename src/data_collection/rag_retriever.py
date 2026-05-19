from __future__ import annotations

import datetime
import math
import os
import re
import sys
import time
import urllib.parse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

try:
    from src.utils.logger import setup_logger
except ImportError:
    try:
        from utils.logger import setup_logger
    except ImportError:
        from logger import setup_logger

_CURRENT_YEAR  = str(datetime.datetime.now().year)
_PREVIOUS_YEAR = str(datetime.datetime.now().year - 1)


INSTITUTION_REGISTRY: dict[str, str] = {
    # === International health & science bodies ===
    r"\bwho\b":             "who.int",
    r"\bworld health organization\b": "who.int",
    r"world health organisation": "who.int",
    r"\bnih\b":             "nih.gov",
    r"national institutes? of health": "nih.gov",
    r"\bcdc\b":             "cdc.gov",
    r"centers? for disease control": "cdc.gov",
    r"\bfda\b":             "fda.gov",
    r"food and drug administration": "fda.gov",
    r"\bema\b":             "ema.europa.eu",
    r"european medicines agency": "ema.europa.eu",
    r"\bipcc\b":            "ipcc.ch",
    r"intergovernmental panel on climate change": "ipcc.ch",
    # === Space & science agencies ===
    r"\bnasa\b":            "nasa.gov",
    r"\besa\b":             "esa.int",
    r"european space agency": "esa.int",
    # === Universities & research institutions ===
    r"\boxford\b":          "ox.ac.uk",
    r"university of oxford": "ox.ac.uk",
    r"\bcambridge\b":       "cam.ac.uk",
    r"university of cambridge": "cam.ac.uk",
    r"\bharvard\b":         "harvard.edu",
    r"\bmit\b":             "mit.edu",
    r"massachusetts institute of technology": "mit.edu",
    r"\bstanford\b":        "stanford.edu",
    r"\bjohns hopkins\b":   "jhu.edu",
    # === Intergovernmental / political bodies ===
    r"\bun\b(?!\w)":        "un.org",
    r"\bunited nations\b":  "un.org",
    r"\bnato\b":            "nato.int",
    r"\bimf\b":             "imf.org",
    r"international monetary fund": "imf.org",
    r"\bworld bank\b":      "worldbank.org",
    r"\bwto\b":             "wto.org",
    r"world trade organization": "wto.org",
    # === National government bodies (US) ===
    r"\bfbi\b":             "fbi.gov",
    r"\bcia\b":             "cia.gov",
    r"\bnoaa\b":            "noaa.gov",
    r"\bnsf\b":             "nsf.gov",
    r"national science foundation": "nsf.gov",
    # === UK government ===
    r"\buk government\b":   "gov.uk",
    r"\bbritish government\b": "gov.uk",
    r"\bprime minister\b":  "gov.uk",
    r"\b10 downing\b":      "gov.uk",
    r"\bhouse of commons\b": "parliament.uk",
    r"\bhouse of lords\b":  "parliament.uk",
    r"\buk parliament\b":   "parliament.uk",
    # === EU institutions ===
    r"\beuropean commission\b": "ec.europa.eu",
    r"\beuropean parliament\b": "europarl.europa.eu",
    r"\beuropean council\b":    "consilium.europa.eu",
}

_INSTITUTION_PATTERN = re.compile(
    "|".join(f"(?P<inst_{i}>{pat})" for i, pat in enumerate(INSTITUTION_REGISTRY)),
    re.IGNORECASE,
)
_INSTITUTION_DOMAINS = list(INSTITUTION_REGISTRY.values())


def _detect_institution(claim: str) -> Optional[str]:
    match = _INSTITUTION_PATTERN.search(claim)
    if not match:
        return None
    for group_name, text in match.groupdict().items():
        if text is not None:
            idx = int(group_name.split("_")[1])
            return _INSTITUTION_DOMAINS[idx]
    return None


def _build_intent_query(claim_keywords: str, claim: str) -> str:
    domain = _detect_institution(claim)
    if domain:
        return f"{claim_keywords} site:{domain}"
    return f"{claim_keywords} official statement OR scientific consensus"

_REFUTATION_TRIGGERS = re.compile(
    r"""
    \b(
        # Medical / biological
        vaccine|vaxx|vaccination|infertility|fertility|cancer|cure|treat|therapy|
        autism|microchip|5g|radiation|toxin|chemical|poison|dna|mrna|spike|
        ivermectin|hydroxychloroquine|bleach|miracle|supplement|detox|
        # Scientific domains
        climate|global.warming|evolution|species|quantum|relativity|gravity|
        virus|bacteria|pathogen|pandemic|epidemic|transmission|mutation|variant|
        study|research|trial|experiment|proof|evidence|scientist|researcher|
        laboratory|lab|peer.review|journal|published|
        # Institutional assertion markers
        confirmed|announced|discovered|proved|proven|found|linked|caused|
        according.to|claimed.by|states.that|says.that
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _is_refutable_claim(claim: str) -> bool:
    return bool(_REFUTATION_TRIGGERS.search(claim))

_POLITICAL_APPOINTMENT_TRIGGERS = re.compile(
    r"""
    \b(
        prime.minister|president|chancellor|minister|secretary|governor|
        senator|congressman|mp\b|member.of.parliament|
        became|appointed|elected|sworn.in|took.office|
        general.election|election|won.*majority|majority.*won|
        cabinet|government|coalition|parliament|congress|senate
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _is_political_appointment_claim(claim: str) -> bool:

    return bool(_POLITICAL_APPOINTMENT_TRIGGERS.search(claim))


def _extract_person_name(claim: str, keywords: list) -> Optional[str]:
    name_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b')
    matches = name_pattern.findall(claim)
    role_words = {
        "Prime Minister", "President", "The United", "United Kingdom",
        "Labour Party", "General Election", "House Of", "Secretary Of",
    }
    for m in matches:
        if m not in role_words and len(m.split()) <= 3:
            return m
    for kw in keywords:
        if kw and kw[0].isupper() and len(kw.split()) <= 3:
            return kw
    return None


def _build_refutation_query(claim_keywords: str, claim: str) -> str:
    claim_lower = claim.lower()

    if re.search(r'\b(5g|wifi|wi-fi|bluetooth|emf|electromagnetic|radiation|frequency|frequencies)\b',
                 claim_lower):
        tech_match = re.search(
            r'\b(5g|wifi|wi-fi|bluetooth|emf|electromagnetic|radiation)\b',
            claim_lower
        )
        tech = tech_match.group(1).upper() if tech_match else claim_keywords.split()[0]
        return f"{tech} health effects scientific consensus WHO fact check"

    if re.search(r'\b(vaccine|vaxx|vaccination|mrna|spike protein|shedding)\b', claim_lower):
        if re.search(r'\b(shed|shedding|transmit|transmission|contact|physical)\b', claim_lower):
            return f"mRNA vaccine spike protein shedding transmission debunked CDC WHO scientific"
        if re.search(r'\b(better|more effective|superior|outperform|replace|prevent)\b', claim_lower):
            return f"{claim_keywords} vs booster clinical trial RCT evidence WHO"
        return f"{claim_keywords} safety scientific evidence WHO CDC peer reviewed"

    if re.search(r'\b(vitamin|supplement|herb|ivermectin|hydroxychloroquine)\b', claim_lower):
        if re.search(r'\b(better|more effective|superior|outperform|replace|prevent|instead)\b', claim_lower):
            return f"{claim_keywords} clinical evidence peer reviewed WHO CDC comparison RCT"

    if re.search(r'\b(cancer|cure|treat|tumor|tumour)\b', claim_lower):
        return f"{claim_keywords} scientific consensus clinical evidence debunked"

    if re.search(r'\b(climate|global warming|evolution|gravity|quantum)\b', claim_lower):
        return f"{claim_keywords} scientific consensus evidence"

    return f"{claim_keywords} fact check scientific consensus evidence"



@dataclass
class RetrievedContext:
    query: str
    keywords: list
    extractor_used: str = "unknown"
    wikipedia_results: list = field(default_factory=list)
    duckduckgo_results: list = field(default_factory=list)
    refutation_results: list = field(default_factory=list)
    combined_context: str = ""
    source_count: int = 0
    institution_domain: Optional[str] = None
    intent_query: str = ""
    refutation_query: str = ""

    def to_prompt_string(self) -> str:
        return self.combined_context

    def is_empty(self) -> bool:
        return self.source_count == 0

    @property
    def institution_named(self) -> bool:
        return self.institution_domain is not None

    @property
    def has_refutation(self) -> bool:
        return len(self.refutation_results) > 0


def _recency_score(item: dict) -> int:
    haystack = (
        (item.get("title") or "") + " " + (item.get("snippet") or "")
    ).lower()
    if _CURRENT_YEAR in haystack:
        return 2
    if _PREVIOUS_YEAR in haystack:
        return 1
    return 0


def _prioritize_recent_snippets(snippets: list) -> list:

    return sorted(snippets, key=_recency_score, reverse=True)


class QueryExtractor(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name used in logs and RetrievedContext.extractor_used."""

    @abstractmethod
    def extract(self, claim: str, max_keywords: int = 4) -> list:
        """Extract search keywords from a claim. Returns [] on failure."""


class OllamaQueryExtractor(QueryExtractor):

    _SYSTEM_PROMPT = (
        "You are a search query expert for a fact-checking system. "
        "Your only job is to extract the 3-4 most important search keywords "
        "or short phrases from a claim. These will be used to retrieve "
        "evidence from Wikipedia and the web.\n\n"
        "Rules:\n"
        "- Preserve named entities exactly (people, places, organisations, dates).\n"
        "- Prefer specific nouns and noun phrases over verbs.\n"
        "- Return ONLY a comma-separated list. No explanation, no numbering.\n\n"
        "Examples:\n"
        "Claim: 'Scientists found a pyramid on Mars in 2024'\n"
        "Output: Mars pyramid, Mars discovery 2024, scientists Mars\n\n"
        "Claim: 'COVID-19 vaccines cause infertility according to a new study'\n"
        "Output: COVID-19 vaccine infertility, vaccine side effects study\n\n"
        "Claim: 'Volodymyr Zelensky announced Ukraine peace talks in Istanbul'\n"
        "Output: Zelensky Ukraine peace talks, Istanbul negotiations 2024"
    )

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model: str = "mistral",
        timeout: int = 15,
    ):
        self.endpoint = f"{ollama_host.rstrip('/')}/api/generate"
        self.model    = model
        self.timeout  = timeout

    @property
    def name(self) -> str:
        return f"ollama/{self.model}"

    def extract(self, claim: str, max_keywords: int = 4) -> list:
        prompt = (
            f"{self._SYSTEM_PROMPT}\n\n"
            f"Claim: '{claim}'\n"
            f"Output (max {max_keywords} items):"
        )
        payload = {
            "model":   self.model,
            "prompt":  prompt,
            "stream":  False,
            "options": {"temperature": 0.0, "num_predict": 80},
        }
        try:
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
            raw_text: str = response.json().get("response", "").strip()
            if not raw_text:
                return []
            return self._parse_response(raw_text, max_keywords)
        except Exception:
            return []

    @staticmethod
    def _parse_response(raw: str, max_keywords: int) -> list:
        normalised = raw.replace("\n", ",").replace(";", ",")
        parts = [p.strip().strip("*-123456789.").strip() for p in normalised.split(",")]
        return [p for p in parts if p and len(p) > 1][:max_keywords]

class SpacyQueryExtractor(QueryExtractor):

    _PRIORITY_LABELS  = {"PERSON", "ORG", "GPE", "LOC", "EVENT", "NORP", "FAC", "PRODUCT"}
    _SECONDARY_LABELS = {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}
    _STOP_WORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "and", "or", "but", "if", "that",
        "this", "these", "those", "it", "its", "not", "no", "so", "as",
        "also", "just", "very", "too", "new", "said", "say", "says",
        "according", "claim", "claims", "report", "reports",
    }

    def __init__(self, model: str = "en_core_web_sm"):
        self.model_name = model
        self._nlp = None

    @property
    def name(self) -> str:
        return f"spacy/{self.model_name}"

    def _load_nlp(self):
        if self._nlp is not None:
            return self._nlp
        try:
            import spacy
            self._nlp = spacy.load(self.model_name)
        except ImportError:
            self._nlp = None
        except OSError:
            try:
                import spacy, subprocess
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", self.model_name],
                    capture_output=True, check=True,
                )
                self._nlp = spacy.load(self.model_name)
            except Exception:
                self._nlp = None
        return self._nlp

    def extract(self, claim: str, max_keywords: int = 4) -> list:
        nlp = self._load_nlp()
        if nlp is None:
            return []
        doc      = nlp(claim)
        keywords: list = []
        seen:     set  = set()
        for ent in doc.ents:
            if ent.label_ in self._PRIORITY_LABELS:
                text = ent.text.strip()
                key  = text.lower()
                if key not in seen and len(text) > 1:
                    keywords.append(text); seen.add(key)
        for ent in doc.ents:
            if ent.label_ in self._SECONDARY_LABELS and len(keywords) < max_keywords:
                text = ent.text.strip(); key = text.lower()
                if key not in seen and len(text) > 1:
                    keywords.append(text); seen.add(key)
        for token in doc:
            if len(keywords) >= max_keywords:
                break
            if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop:
                key = token.lemma_.lower()
                if key not in seen and key not in self._STOP_WORDS and len(key) > 2:
                    keywords.append(token.text); seen.add(key)
        return keywords[:max_keywords]


class HeuristicQueryExtractor(QueryExtractor):

    _STOP_WORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "up",
        "about", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again", "further",
        "then", "once", "and", "or", "but", "if", "while", "although", "that",
        "this", "these", "those", "it", "its", "we", "they", "he", "she",
        "i", "you", "not", "no", "so", "as", "also", "than", "just", "more",
        "very", "too", "such", "both", "each", "few", "other", "some",
        "what", "which", "who", "when", "where", "why", "how", "new",
        "said", "say", "says", "according", "claim", "claims",
    }

    @property
    def name(self) -> str:
        return "heuristic"

    def extract(self, claim: str, max_keywords: int = 4) -> list:
        tokens = re.findall(r'\b[a-zA-Z]{3,}\b', claim)
        n      = len(tokens)
        scores: dict = {}
        for idx, token in enumerate(tokens):
            lower = token.lower()
            if lower in self._STOP_WORDS:
                continue
            position_boost = 1.3 if idx < n / 2 else 1.0
            base = math.log(len(lower) + 1)
            scores[lower] = scores.get(lower, 0.0) + base * position_boost
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in ranked[:max_keywords]]

class RAGRetriever:
    WIKIPEDIA_API  = "https://en.wikipedia.org/w/api.php"
    DDG_TIME_LIMIT = "y"

    def __init__(
        self,
        config_path: Optional[str] = None,
        max_keywords: int = 4,
        max_wiki_results: int = 2,
        max_ddg_results: int = 3,
        max_refutation_results: int = 2,
        timeout: int = 10,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        ollama_host: str = "http://localhost:11434",
        ollama_model: str = "mistral",
        ollama_timeout: int = 15,
        spacy_model: str = "en_core_web_sm",
        ddg_time_limit: str = "y",
    ):
        self.logger = setup_logger("RAGRetriever", config_path=config_path)

        if config_path:
            self._load_config(config_path)
        else:
            self.max_keywords           = max_keywords
            self.max_wiki_results       = max_wiki_results
            self.max_ddg_results        = max_ddg_results
            self.max_refutation_results = max_refutation_results
            self.timeout                = timeout
            self.max_retries            = max_retries
            self.retry_delay            = retry_delay
            self.ollama_host            = ollama_host
            self.ollama_model           = ollama_model
            self.ollama_timeout         = ollama_timeout
            self.spacy_model            = spacy_model
            self.ddg_time_limit         = ddg_time_limit

        self._extractors: list = [
            OllamaQueryExtractor(
                ollama_host=self.ollama_host,
                model=self.ollama_model,
                timeout=self.ollama_timeout,
            ),
            SpacyQueryExtractor(model=self.spacy_model),
            HeuristicQueryExtractor(),
        ]

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "FakeNewsDetector/1.0 (academic research; "
                "contact: admin@example.com)"
            )
        })

        self.logger.info(
            "RAGRetriever ready | extractors=[%s] | "
            "wiki=%d  ddg=%d  refutation=%d  keywords=%d  ddg_timelimit=%s",
            " -> ".join(e.name for e in self._extractors),
            self.max_wiki_results, self.max_ddg_results,
            self.max_refutation_results, self.max_keywords,
            getattr(self, "ddg_time_limit", self.DDG_TIME_LIMIT),
        )

    def _load_config(self, config_path: str) -> None:
        import yaml
        defaults = dict(
            max_keywords=4, max_wiki_results=3, max_ddg_results=3,
            max_refutation_results=2, timeout=15, max_retries=3,
            retry_delay=1.0, ollama_host="http://localhost:11434",
            ollama_model="mistral", ollama_timeout=20,
            spacy_model="en_core_web_sm", ddg_time_limit="y",
        )
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            r = cfg.get("retriever", {})
            for key, default in defaults.items():
                setattr(self, key, r.get(key, default))
            self.logger.info("Retriever config loaded from '%s'.", config_path)
        except Exception as exc:
            self.logger.warning("Config not loaded (%s). Using defaults.", exc)
            for key, default in defaults.items():
                setattr(self, key, default)

    def _extract_keywords(self, claim: str) -> tuple:
        for extractor in self._extractors:
            try:
                keywords = extractor.extract(claim, max_keywords=self.max_keywords)
                if keywords:
                    self.logger.info("Keywords via [%s]: %s", extractor.name, keywords)
                    return keywords, extractor.name
                self.logger.debug("[%s] returned no keywords, trying next.", extractor.name)
            except Exception as exc:
                self.logger.warning("[%s] error: %s -- trying next.", extractor.name, exc)
        self.logger.error("All extractors failed.")
        return [], "none"

    def _get(self, url: str, params: dict) -> Optional[dict]:
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.Timeout:
                self.logger.warning("Timeout attempt %d/%d: %s", attempt, self.max_retries, url)
            except requests.exceptions.HTTPError as exc:
                self.logger.warning("HTTP %s attempt %d: %s", exc.response.status_code, attempt, url)
            except requests.exceptions.ConnectionError:
                self.logger.warning("Connection error attempt %d/%d.", attempt, self.max_retries)
            except ValueError:
                self.logger.error("JSON parse error: %s", url)
                return None
            if attempt < self.max_retries:
                time.sleep(self.retry_delay * (2 ** (attempt - 1)))
        self.logger.error("All %d attempts failed: %s", self.max_retries, url)
        return None

    def _search_wikipedia(self, query: str) -> list:
        """Query MediaWiki API and return page extracts, re-ranked by recency."""
        self.logger.debug("[Wikipedia] Searching: '%s'", query)

        search_data = self._get(self.WIKIPEDIA_API, {
            "action": "query", "list": "search",
            "srsearch": query, "srlimit": self.max_wiki_results,
            "format": "json", "utf8": 1,
        })
        if not search_data:
            return []

        search_results = search_data.get("query", {}).get("search", [])
        if not search_results:
            return []

        page_titles  = [r["title"] for r in search_results]
        extract_data = self._get(self.WIKIPEDIA_API, {
            "action": "query",
            "titles": "|".join(page_titles),
            "prop": "extracts|info",
            "exintro": True, "explaintext": True,
            "exsentences": 10, "inprop": "url",
            "format": "json", "utf8": 1,
        })
        if not extract_data:
            return []

        results = []
        pages = extract_data.get("query", {}).get("pages", {})
        for page in pages.values():
            if page.get("ns") != 0 or "missing" in page:
                continue
            title   = page.get("title", "")
            extract = page.get("extract", "").strip()
            url = page.get(
                "fullurl",
                f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}",
            )
            if extract:
                results.append({
                    "source": "wikipedia", "title": title,
                    "snippet": extract[:1200], "url": url,
                })

        return _prioritize_recent_snippets(results)

    def _fetch_wikipedia_precise_date(self, title: str) -> Optional[dict]:

        premiership_title = f"Premiership of {title}"
        for try_title in [premiership_title, title]:
            data = self._get(self.WIKIPEDIA_API, {
                "action": "query",
                "titles": try_title,
                "prop": "extracts|info",
                "exintro": True, "explaintext": True,
                "exsentences": 20, "inprop": "url",
                "format": "json", "utf8": 1,
            })
            if not data:
                continue
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                if page.get("ns") != 0 or "missing" in page:
                    continue
                extract = page.get("extract", "").strip()
                if not extract:
                    continue
                title_out = page.get("title", try_title)
                url = page.get(
                    "fullurl",
                    f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title_out)}",
                )
                return {
                    "source": "wikipedia",
                    "title": title_out,
                    "snippet": extract[:2000],
                    "url": url,
                }
        return None

    def _search_duckduckgo(
        self,
        query: str,
        timelimit: Optional[str] = None,
        max_results: Optional[int] = None,
        topic_keywords: Optional[list] = None,
    ) -> list:

        try:
            from ddgs import DDGS
        except ImportError:
            self.logger.error(
                "duckduckgo_search is not installed. "
                "Run: pip install duckduckgo-search"
            )
            return []

        effective_timelimit = (
            timelimit if timelimit is not None
            else getattr(self, "ddg_time_limit", self.DDG_TIME_LIMIT)
        )
        effective_max = (
            max_results if max_results is not None else self.max_ddg_results
        )

        self.logger.debug(
            "[DuckDuckGo] query='%s' timelimit=%s max=%d",
            query, effective_timelimit, effective_max,
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                with DDGS() as ddgs:
                    raw = list(ddgs.text(
                        query,
                        max_results=effective_max,
                        timelimit=effective_timelimit,
                    ))

                results = self._parse_ddg_raw(raw, topic_keywords)

                # Fallback: timelimit too strict -> retry without restriction
                if not results and effective_timelimit:
                    self.logger.warning(
                        "[DuckDuckGo] timelimit='%s' returned 0 results for '%s'. "
                        "Retrying without time restriction.",
                        effective_timelimit, query,
                    )
                    with DDGS() as ddgs_fb:
                        raw_fb = list(ddgs_fb.text(query, max_results=effective_max))
                    results = self._parse_ddg_raw(raw_fb, topic_keywords)

                self.logger.debug("[DuckDuckGo] %d results for '%s'.", len(results), query)
                return _prioritize_recent_snippets(results)

            except Exception as exc:
                self.logger.warning(
                    "[DuckDuckGo] Attempt %d/%d failed: %s", attempt, self.max_retries, exc
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2 ** (attempt - 1)))

        self.logger.error("[DuckDuckGo] All attempts failed for: '%s'", query)
        return []

    @staticmethod
    def _parse_ddg_raw(raw: list, topic_keywords: Optional[list] = None) -> list:

        results = []
        kw_lower = [k.lower() for k in (topic_keywords or [])]

        for item in raw:
            title   = (item.get("title") or "").strip()
            snippet = (item.get("body") or "").strip()
            url     = (item.get("href") or "").strip()
            if not snippet or len(snippet) < 30:
                continue

            # Pre-filter: if topic keywords given, at least one must appear
            if kw_lower:
                haystack = (title + " " + snippet).lower()
                if not any(kw in haystack for kw in kw_lower):
                    continue   # drift detected -- skip before sending to LLM

            results.append({
                "source":  "duckduckgo_web",
                "title":   title[:120],
                "snippet": snippet[:600],
                "url":     url,
            })
        return results

    def _build_context_string(self, context: RetrievedContext) -> str:

        if (not context.wikipedia_results
                and not context.duckduckgo_results
                and not context.refutation_results):
            return ""

        import datetime as _dt
        _retrieval_date = _dt.datetime.utcnow().strftime("%Y-%m-%d")

        lines = [
            "=== RETRIEVED CONTEXT ===\n",
            f"[RETRIEVAL DATE: {_retrieval_date} UTC -- This evidence was fetched "
            f"in real-time today. Wikipedia and Tier 1 sources here OVERRIDE "
            f"any training-data knowledge about the same topic. "
            f"Specific dates in snippets (e.g. election results, PM appointments, "
            f"WHO declarations) are more current than your training cutoff.]\n",
        ]

        # Search strategy annotation
        if context.institution_domain:
            lines.append(
                f"[Search strategy: TARGETED -- site:{context.institution_domain}]\n"
            )
        else:
            lines.append(
                "[Search strategy: BROAD INTENT -- official statement / scientific consensus]\n"
            )

        # Wikipedia
        for item in context.wikipedia_results:
            # Flag snippets that mention the current or previous year
            recency_flag = ""
            haystack = (item.get("title", "") + " " + item.get("snippet", "")).lower()
            if _CURRENT_YEAR in haystack:
                recency_flag = f" [CONTAINS {_CURRENT_YEAR} DATA -- MAY POSTDATE TRAINING CUTOFF]"
            elif _PREVIOUS_YEAR in haystack:
                recency_flag = f" [CONTAINS {_PREVIOUS_YEAR} DATA -- MAY POSTDATE TRAINING CUTOFF]"
            lines.append(f"[Wikipedia] {item['title']}{recency_flag}")
            lines.append(item["snippet"])
            lines.append(f"Source: {item['url']}\n")

        # Primary DuckDuckGo
        for item in context.duckduckgo_results:
            lines.append(f"[DuckDuckGo] {item['title']}")
            lines.append(item["snippet"])
            if item.get("url"):
                lines.append(f"Source: {item['url']}\n")

        # Refutation branch -- clearly delimited
        if context.refutation_results:
            lines.append(
                "\n=== REFUTATION SEARCH RESULTS ===\n"
                "[IMPORTANT: The following snippets were retrieved by a dedicated search "
                "for active counter-evidence: scientific consensus against the claim, "
                "direct rebuttals, or fact-check debunking. These represent potential "
                "EVIDENCE OF ABSENCE, not merely lack of support. Treat them with "
                "higher evidentiary weight than a simple failure to find the original claim.]\n"
            )
            lines.append(f"[Refutation query used: '{context.refutation_query}']\n")
            for item in context.refutation_results:
                lines.append(f"[REFUTATION] {item['title']}")
                lines.append(item["snippet"])
                if item.get("url"):
                    lines.append(f"Source: {item['url']}\n")

        return "\n".join(lines)

    # -- Public API ------------------------------------------------------------

    def retrieve(self, text: str) -> RetrievedContext:

        preview = text[:100] + ("..." if len(text) > 100 else "")
        self.logger.info("Retrieving context for: '%s'", preview)

        keywords, extractor_used = self._extract_keywords(text)
        if not keywords:
            self.logger.warning("No keywords extracted. Returning empty context.")
            return RetrievedContext(
                query=text, keywords=[], extractor_used="none", combined_context=""
            )

        primary_query      = " ".join(keywords[:4])
        institution_domain = _detect_institution(text)
        intent_query       = _build_intent_query(primary_query, text)

        self.logger.info(
            "[Intent] %s | DDG query: '%s'",
            f"Institution detected: domain='{institution_domain}'"
            if institution_domain else "No institution found. Broad-intent",
            intent_query,
        )

        # -- Primary searches --------------------------------------------------
        wiki_results = self._search_wikipedia(primary_query)
        ddg_results  = self._search_duckduckgo(intent_query)

        if _is_political_appointment_claim(text):
            person_name = _extract_person_name(text, keywords)
            if person_name:
                self.logger.info(
                    "[PoliticalAppointment] Claim detected. Fetching precise "
                    "Wikipedia extract for: '%s'", person_name
                )
                precise = self._fetch_wikipedia_precise_date(person_name)
                if precise:
                    # Replace or enrich wiki_results with the precise extract
                    # Put it first so Mistral sees it with highest priority
                    existing_titles = {r["title"].lower() for r in wiki_results}
                    if precise["title"].lower() not in existing_titles:
                        wiki_results.insert(0, precise)
                    else:
                        # Replace the shorter version with the longer one
                        wiki_results = [
                            precise if r["title"].lower() == precise["title"].lower()
                            else r for r in wiki_results
                        ]
                    self.logger.info(
                        "[PoliticalAppointment] Precise extract fetched: "
                        "'%s' (%d chars)", precise["title"], len(precise["snippet"])
                    )
            else:
                self.logger.info(
                    "[PoliticalAppointment] Could not extract person name from claim."
                )

        _who_covid_pattern = re.compile(
            r'\b(who|world health organization|world health organisation)\b.*\b(covid|coronavirus|pheic|health emergency)\b'
            r'|\b(covid|coronavirus|pheic|health emergency)\b.*\b(who|world health organization)\b',
            re.IGNORECASE
        )
        if _who_covid_pattern.search(text):
            self.logger.info(
                "[WHODeclaration] WHO+COVID claim detected. Force-fetching "
                "'COVID-19 pandemic' Wikipedia article."
            )
            covid_article = self._fetch_wikipedia_precise_date("COVID-19 pandemic")
            if covid_article:
                existing_titles = {r["title"].lower() for r in wiki_results}
                if covid_article["title"].lower() not in existing_titles:
                    wiki_results.insert(0, covid_article)
                    self.logger.info(
                        "[WHODeclaration] Injected '%s' (%d chars) into wiki_results.",
                        covid_article["title"], len(covid_article["snippet"])
                    )


        refutation_results = []
        refutation_query   = ""

        if _is_refutable_claim(text):
            refutation_query = _build_refutation_query(primary_query, text)
            self.logger.info(
                "[Refutation] Claim is refutable. Firing: '%s'", refutation_query
            )
            refutation_results = self._search_duckduckgo(
                refutation_query,
                max_results=self.max_refutation_results,
                topic_keywords=keywords,        # enables retriever-level drift filter
            )
            self.logger.info(
                "[Refutation] Retrieved %d snippet(s).", len(refutation_results)
            )
        else:
            self.logger.info(
                "[Refutation] Claim not flagged as health/scientific -- branch skipped."
            )

        # -- Fallback: all primary branches empty, retry with single keyword ---
        if not wiki_results and not ddg_results and len(keywords) > 1:
            fallback_query        = keywords[0]
            fallback_intent_query = _build_intent_query(fallback_query, text)
            self.logger.info(
                "Primary returned nothing. Fallback: '%s' | DDG: '%s'",
                fallback_query, fallback_intent_query,
            )
            wiki_results = self._search_wikipedia(fallback_query)
            ddg_results  = self._search_duckduckgo(fallback_intent_query)
            intent_query = fallback_intent_query

            if _is_refutable_claim(text) and not refutation_results:
                refutation_query   = _build_refutation_query(fallback_query, text)
                refutation_results = self._search_duckduckgo(
                    refutation_query,
                    max_results=self.max_refutation_results,
                    topic_keywords=keywords,
                )

        total = len(wiki_results) + len(ddg_results) + len(refutation_results)

        context = RetrievedContext(
            query=text,
            keywords=keywords,
            extractor_used=extractor_used,
            wikipedia_results=wiki_results,
            duckduckgo_results=ddg_results,
            refutation_results=refutation_results,
            source_count=total,
            institution_domain=institution_domain,
            intent_query=intent_query,
            refutation_query=refutation_query,
        )
        context.combined_context = self._build_context_string(context)

        self.logger.info(
            "Retrieved %d wiki + %d ddg + %d refutation = %d total snippets "
            "(%d chars) via [%s] | intent=%s | has_refutation=%s.",
            len(wiki_results), len(ddg_results), len(refutation_results), total,
            len(context.combined_context), extractor_used,
            "targeted" if institution_domain else "broad",
            context.has_refutation,
        )
        return context

    def retrieve_by_keywords(self, keywords: list) -> RetrievedContext:
        """Retrieve context from a pre-built keyword list (bypasses extraction)."""
        self.logger.info("retrieve_by_keywords called with: %s", keywords)
        return self._retrieve_with_keywords(" ".join(keywords), keywords, "external")

    def _retrieve_with_keywords(
        self, original_text: str, keywords: list, extractor_name: str
    ) -> RetrievedContext:
        primary_query      = " ".join(keywords[:4])
        institution_domain = _detect_institution(original_text)
        intent_query       = _build_intent_query(primary_query, original_text)

        wiki_results  = self._search_wikipedia(primary_query)
        ddg_results   = self._search_duckduckgo(intent_query)

        refutation_results = []
        refutation_query   = ""
        if _is_refutable_claim(original_text):
            refutation_query   = _build_refutation_query(primary_query, original_text)
            refutation_results = self._search_duckduckgo(
                refutation_query, max_results=self.max_refutation_results
            )

        total = len(wiki_results) + len(ddg_results) + len(refutation_results)
        context = RetrievedContext(
            query=original_text,
            keywords=keywords,
            extractor_used=extractor_name,
            wikipedia_results=wiki_results,
            duckduckgo_results=ddg_results,
            refutation_results=refutation_results,
            source_count=total,
            institution_domain=institution_domain,
            intent_query=intent_query,
            refutation_query=refutation_query,
        )
        context.combined_context = self._build_context_string(context)
        return context

if __name__ == "__main__":
    TEST_CLAIMS = [
        "Scientists found a pyramid on Mars in 2024",
        "COVID-19 vaccines cause infertility according to new WHO study",
        "Sweden has officially become the 32nd member of NATO in March 2024",
        "Oxford researchers proved coffee cures cancer",
        "NASA confirmed alien life exists on Europa",
    ]
    retriever = RAGRetriever()
    for claim in TEST_CLAIMS:
        print("\n" + "=" * 70)
        print(f"CLAIM          : {claim}")
        ctx = retriever.retrieve(claim)
        print(f"VIA            : {ctx.extractor_used}")
        print(f"KEYS           : {ctx.keywords}")
        print(f"INSTITUTION    : {ctx.institution_domain or '(none)'}")
        print(f"PRIMARY QUERY  : {ctx.intent_query}")
        print(f"REFUTATION Q   : {ctx.refutation_query or '(skipped)'}")
        print(
            f"SRCS           : {ctx.source_count} total "
            f"({len(ctx.wikipedia_results)} wiki / "
            f"{len(ctx.duckduckgo_results)} ddg / "
            f"{len(ctx.refutation_results)} refutation)"
        )
        print(f"HAS REFUTATION : {ctx.has_refutation}")
        if ctx.combined_context:
            print("\n" + ctx.combined_context[:900] + "...")
        else:
            print("(no context retrieved)")