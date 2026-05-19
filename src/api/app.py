import json
import logging
import os
import sys
import time
from flask_cors import CORS
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from flask import Flask, jsonify, request

try:
    from src.utils.logger import setup_logger
    from src.models.roberta_model import RoBERTaModel
    from src.data_collection.rag_retriever import RAGRetriever
    from src.counter_narrative.generator import CounterNarrativeGenerator
    from src.models.sentiment_model import SentimentAnalyzer
except ImportError as e:
    print(f"[API] Import error: {e}")
    print("[API] Make sure you are running from the project root directory.")
    sys.exit(1)

PIPELINE: dict = {}

CONFIG_PATH = "configs/config.yaml"


def _load_pipeline() -> None:
    logger = setup_logger("API", config_path=CONFIG_PATH)
    logger.info("Loading pipeline components (this runs once at startup)...")

    t0 = time.time()

    roberta = RoBERTaModel(config_path=CONFIG_PATH)
    roberta.model_settings["name"] = "models/roberta/acc_8183/checkpoint-5100"
    roberta.load_model()

    retriever  = RAGRetriever(config_path=CONFIG_PATH)
    generator  = CounterNarrativeGenerator(config_path=CONFIG_PATH)
    sentiment  = SentimentAnalyzer(config_path=CONFIG_PATH)

    PIPELINE["roberta"]            = roberta
    PIPELINE["retriever"]          = retriever
    PIPELINE["generator"]          = generator
    PIPELINE["sentiment_analyzer"] = sentiment
    PIPELINE["logger"]             = logger

    elapsed = time.time() - t0
    logger.info("Pipeline ready in %.1fs. API is accepting requests.", elapsed)

def run_pipeline(claim: str) -> dict:
    if not PIPELINE:
        return {"error": "Pipeline not initialised. Call _load_pipeline() first."}

    roberta  = PIPELINE["roberta"]
    retriever = PIPELINE["retriever"]
    generator = PIPELINE["generator"]
    analyzer  = PIPELINE["sentiment_analyzer"]
    logger    = PIPELINE.get("logger", logging.getLogger("API"))

    logger.info("Processing claim: '%s'", claim[:80])

    try:
        roberta_result = roberta.predict(claim)

        context = retriever.retrieve(claim)

        sentiment_result = analyzer.analyze(claim)

        narrative_result = generator.generate(
            claim=claim,
            roberta_result=roberta_result,
            context=context,
            sentiment_result=sentiment_result,
        )

        return {
            "claim": claim,
            "roberta": {
                "label":      roberta_result.label,
                "confidence": round(roberta_result.confidence, 4),
            },
            "final_verdict": {
                "label":         narrative_result.final_label,
                "is_correction": narrative_result.is_correction,
                "explanation":   narrative_result.explanation,
            },
            "counter_narrative": narrative_result.counter_narrative,
            "sources":           narrative_result.sources_used,
            "manipulation_analysis": {
                "primary_emotion":    sentiment_result.primary_emotion,
                "confidence":         round(sentiment_result.primary_confidence, 4),
                "subjectivity_score": round(sentiment_result.subjectivity_score, 4),
                "manipulation_risk":  sentiment_result.manipulation_risk,
                "risk_reason":        sentiment_result.manipulation_reason,
                "linguistic_override": sentiment_result.linguistic_override,
            },
        }

    except Exception as exc:
        logger.exception("Pipeline error for claim '%s': %s", claim[:80], exc)
        return {"error": str(exc), "claim": claim}

from flask import Flask, jsonify, request, render_template  # Համոզվիր, որ սա ավելացրել ես վերևում


def create_app() -> Flask:
    if not PIPELINE:
        _load_pipeline()

    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))

    app = Flask(__name__, template_folder=template_dir)
    CORS(app, resources={r"/*": {"origins": "*"}}, methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type"])
    app.config["JSON_ENSURE_ASCII"] = False

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "pipeline_loaded": bool(PIPELINE),
        })

    @app.route("/verify", methods=["POST"])
    def verify():
        body = request.get_json(silent=True)
        if not body:
            return jsonify({"error": "Request body must be JSON."}), 400

        claim = (body.get("claim") or "").strip()
        if not claim:
            return jsonify({"error": "'claim' field is required and cannot be empty."}), 400

        t0 = time.time()
        result = run_pipeline(claim)
        elapsed_ms = round((time.time() - t0) * 1000)

        if "error" in result and "claim" not in result:
            return jsonify(result), 500

        result["processing_time_ms"] = elapsed_ms
        return jsonify(result), 200

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False)