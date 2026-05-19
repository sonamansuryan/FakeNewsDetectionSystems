from flask_cors import CORS

CORS(
    app,
    resources={r"/verify": {"origins": "*"}, r"/health": {"origins": "*"}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# def create_app() -> Flask:
#     if not PIPELINE:
#         _load_pipeline()
#
#     template_dir = os.path.abspath(
#         os.path.join(os.path.dirname(__file__), 'templates')
#     )
#     app = Flask(__name__, template_folder=template_dir)
#     app.config["JSON_ENSURE_ASCII"] = False
#
#     # ── CORS (Chrome Extension-ի համար) ──────────────────────────────────
#     CORS(
#         app,
#         resources={r"/verify": {"origins": "*"}, r"/health": {"origins": "*"}},
#         methods=["GET", "POST", "OPTIONS"],
#         allow_headers=["Content-Type"],
#     )
#
#     @app.route("/", methods=["GET"])
#     def index():
#         return render_template("index.html")
#
#     @app.route("/health", methods=["GET"])
#     def health():
#         return jsonify({"status": "ok", "pipeline_loaded": bool(PIPELINE)})
#
#     @app.route("/verify", methods=["POST"])
#     def verify():
#         body = request.get_json(silent=True)
#         if not body:
#             return jsonify({"error": "Request body must be JSON."}), 400
#         claim = (body.get("claim") or "").strip()
#         if not claim:
#             return jsonify({"error": "'claim' field is required."}), 400
#         t0 = time.time()
#         result = run_pipeline(claim)
#         result["processing_time_ms"] = round((time.time() - t0) * 1000)
#         if "error" in result and "claim" not in result:
#             return jsonify(result), 500
#         return jsonify(result), 200
#
#     return app