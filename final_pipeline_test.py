import json
import sys
import os
import torch

sys.path.append(os.getcwd())

from src.models.roberta_model import RoBERTaModel
from src.data_collection.rag_retriever import RAGRetriever
from src.counter_narrative.generator import CounterNarrativeGenerator
from src.models.sentiment_model import SentimentAnalyzer


def initialize_pipeline(config_path: str = "configs/config.yaml") -> dict:

    print("--- Initializing Full AI Pipeline (Steps 1-6, Triple-Signal) ---")

    # Step 1 model
    roberta = RoBERTaModel(config_path=config_path)
    roberta.model_settings['name'] = "models/roberta/acc_8183/checkpoint-5100"
    roberta.load_model()

    # Step 2 retriever
    retriever = RAGRetriever(config_path=config_path)

    # Steps 4-6 generator (receives all three signals)
    generator = CounterNarrativeGenerator(config_path=config_path)

    # Step 3 analyzer (runs BEFORE generator so sentiment is ready for prompt)
    sentiment_analyzer = SentimentAnalyzer(config_path=config_path)

    return {
        "roberta":            roberta,
        "retriever":          retriever,
        "generator":          generator,
        "sentiment_analyzer": sentiment_analyzer,
    }


def process_claim(claim: str, components: dict) -> dict:
    roberta   = components["roberta"]
    retriever = components["retriever"]
    generator = components["generator"]
    analyzer  = components["sentiment_analyzer"]

    # ── Step 1: RoBERTa classification ────────────────────────────────────────
    print(f"\n[STEP 1] RoBERTa is analyzing: \"{claim[:60]}...\"")
    roberta_result = roberta.predict(claim)
    print(f"         Result: {roberta_result.label} ({roberta_result.confidence:.2%})")

    # ── Step 2: RAG retrieval ─────────────────────────────────────────────────
    print("[STEP 2] RAG is searching for evidence...")
    context = retriever.retrieve(claim)
    print(f"         Sources found: {context.source_count if context else 0}")

    # ── Step 3: Sentiment analysis (RUNS BEFORE generator) ───────────────────
    # Must run here so sentiment_result is ready to pass into generate().
    # Mistral needs all three signals at prompt-build time.
    print("[STEP 3] SentimentAnalyzer is scoring manipulation risk...")
    sentiment_result = analyzer.analyze(claim)
    print(
        f"         Emotion: {sentiment_result.primary_emotion.upper()} "
        f"({sentiment_result.primary_confidence:.0%}) | "
        f"Subjectivity: {sentiment_result.subjectivity_score:.2f} | "
        f"Risk: {sentiment_result.manipulation_risk}"
    )
    if sentiment_result.linguistic_override:
        print(f"         Linguistic override: {sentiment_result.linguistic_override[:80]}...")

    # ── Steps 4-6: Mistral (Supreme Judge) with all three signals ─────────────
    print("[STEPS 4-6] Mistral is generating final verdict + explanation...")
    narrative_result = generator.generate(
        claim=claim,
        roberta_result=roberta_result,
        context=context,
        sentiment_result=sentiment_result,   # <-- new: sentiment passed in
    )

    return {
        "claim": claim,
        "roberta": {
            "label":      roberta_result.label,
            "confidence": roberta_result.confidence,
        },
        "final_verdict": {
            "label":        narrative_result.final_label,
            "is_correction": narrative_result.is_correction,
            "explanation":  narrative_result.explanation,
        },
        "counter_narrative": narrative_result.counter_narrative,
        "sources":           narrative_result.sources_used,
        "manipulation_analysis": {
            "primary_emotion":    sentiment_result.primary_emotion,
            "confidence":         sentiment_result.primary_confidence,
            "subjectivity_score": sentiment_result.subjectivity_score,
            "manipulation_risk":  sentiment_result.manipulation_risk,
            "risk_reason":        sentiment_result.manipulation_reason,
            "linguistic_override": sentiment_result.linguistic_override,
            "top_emotions":       sentiment_result.top_emotions,
        },
    }


def print_result(result: dict) -> None:
    """Pretty-print the full pipeline result for one claim."""
    manip = result["manipulation_analysis"]

    risk_label = {
        "HIGH":   "HIGH RISK",
        "MEDIUM": "MEDIUM RISK",
        "LOW":    "LOW RISK",
    }.get(manip["manipulation_risk"], manip["manipulation_risk"])

    risk_icon = {
        "HIGH":   "[!!!]",
        "MEDIUM": "[!!] ",
        "LOW":    "[ok] ",
    }.get(manip["manipulation_risk"], "    ")

    print("\n" + "=" * 70)
    print(f"CLAIM          : {result['claim']}")
    print(f"ROBERTA INITIAL: {result['roberta']['label']} "
          f"({result['roberta']['confidence']:.2%})")
    print(f"FINAL VERDICT  : {result['final_verdict']['label']}")

    if result["final_verdict"]["is_correction"]:
        print("*** NOTICE: Mistral overrode RoBERTa's decision based on RAG + Sentiment evidence!")

    print(f"\n--- MANIPULATION ANALYSIS ---")
    print(f"Dominant Emotion : {manip['primary_emotion'].upper()} "
          f"({manip['confidence']:.0%})")
    print(f"Subjectivity     : {manip['subjectivity_score']:.3f} (0=Fact, 1=Opinion)")
    print(f"Manipulation Risk: {risk_icon} {risk_label}")
    print(f"Risk Reason      : {manip['risk_reason']}")
    if manip.get("linguistic_override"):
        print(f"Linguistic Flag  : {manip['linguistic_override']}")

    print(f"\n--- AI EXPLANATION (includes tone analysis) ---")
    print(result["final_verdict"]["explanation"])

    print(f"\n--- COUNTER-NARRATIVE ---")
    print(result["counter_narrative"])
    print("=" * 70)

if __name__ == "__main__":
    CLAIMS = [
        "Drinking alkaline water can cure Stage 1 lung cancer",
        "Sweden has officially become the 32nd member of NATO in March 2024",
        "Scientists PROVE vaccines cause autism! The government is HIDING this from you!",
    ]

    components = initialize_pipeline()

    for claim in CLAIMS:
        res = process_claim(claim, components)
        print_result(res)