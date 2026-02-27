"""
NoDelulu AI — Flask API Server
Supports both default document.txt and user-uploaded documents via the UI.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import tempfile
import traceback

from rag import SimpleRAG
from features import extract_features
from llm_api import call_llm

app = Flask(__name__, static_folder="static")
CORS(app)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DOC_PATH   = os.path.join(BASE_DIR, "document.txt")
MODEL_PATH = os.path.join(BASE_DIR, "rl_guard_policy")

# Load default RAG + PPO model at startup
print("Loading default RAG...")
default_rag = SimpleRAG(DOC_PATH)
print("RAG ready.")

print("Loading RL policy...")
from stable_baselines3 import PPO
rl_model = PPO.load(MODEL_PATH)
print("RL policy ready. Server starting...\n")

ACTION_NAMES  = {0: "ACCEPT", 1: "REGENERATE", 2: "REJECT"}
ACTION_BADGES = {0: "accept", 1: "regen",       2: "reject"}


def run_pipeline(query: str, rag: SimpleRAG):
    """Core RAG → LLM → Features → RL pipeline."""
    # 1. Retrieve context
    chunks = rag.retrieve(query)
    context = "\n".join(chunks) if isinstance(chunks, list) else chunks

    # 2. Generate initial answer
    prompt = f"Context:\n{context}\n\nQuestion:\n{query}\nAnswer:"
    raw_answer = call_llm(prompt)

    # 3. Extract features
    state = extract_features(raw_answer, context, query)

    # 4. RL guard decision
    action = int(rl_model.predict(state)[0])

    # 5. Apply action
    if action == 1:
        final_answer = call_llm(prompt, temperature=0.4)
        state = extract_features(final_answer, context, query)
    elif action == 2:
        final_answer = "The RL guard layer rejected this response due to low confidence and poor document grounding."
    else:
        final_answer = raw_answer

    doc_sim    = float(state[2])
    web_sim    = float(state[3])
    confidence = round(min((doc_sim + web_sim) / 2 * 100, 99), 1)

    return {
        "answer":      final_answer,
        "raw_answer":  raw_answer,
        "action":      ACTION_NAMES[action],
        "badge":       ACTION_BADGES[action],
        "doc_sim":     round(doc_sim, 2),
        "web_sim":     round(web_sim, 2),
        "confidence":  f"{confidence}%",
        "features": {
            "length":      int(state[0]),
            "uncertainty": int(state[1]),
            "doc_sim":     round(float(state[2]), 3),
            "web_sim":     round(float(state[3]), 3),
        }
    }


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data  = request.get_json(force=True)
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    doc_text = data.get("doc_text", "").strip()

    try:
        if doc_text:
            # Build a temporary RAG from the uploaded document text
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                             delete=False, encoding="utf-8") as tmp:
                tmp.write(doc_text)
                tmp_path = tmp.name
            try:
                rag = SimpleRAG(tmp_path)
            finally:
                os.unlink(tmp_path)
        else:
            rag = default_rag

        result = run_pipeline(query, rag)
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "gpt-4o-mini", "rl": "PPO"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)