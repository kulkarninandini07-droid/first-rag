from flask import Flask, render_template, request, jsonify, session
from sentence_transformers import SentenceTransformer
from retriever import load_index_and_chunks, retrieve
from generator import generate_answer
from collections import Counter
from generator import generate_answer, is_greeting, greeting_response
app = Flask(__name__)
app.secret_key = "dev-secret-key-change-later"  # needed for sessions

print("🔧 Loading model and index (one-time startup)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
index, chunks = load_index_and_chunks()
print(f"✅ Ready! {len(chunks)} chunks loaded.\n")

ROLE_LABELS = {
    "student": "Student",
    "finance": "Finance Professional",
    "other": "Non-Finance Background"
}

@app.route("/")
def login():
    return render_template("login.html")

@app.route("/chat")
def chat():
    name = request.args.get("name", "Guest")
    role = request.args.get("role", "other")
    session["name"] = name
    session["role"] = role
    return render_template("index.html", name=name, role=ROLE_LABELS.get(role, "Guest"))

@app.route("/companies")
def companies():
    counts = Counter(c["company"] for c in chunks)
    return jsonify([{"name": name, "chunks": count} for name, count in counts.items()])

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    name = session.get("name", "there")
    role = session.get("role", "other")

    if is_greeting(query):
        return jsonify({"answer": greeting_response(name, role), "sources": []})

    results = retrieve(query, model, index, chunks, k=8)
    answer = generate_answer(query, results, role)
    sources = list(set(f"{r['company']} (page {r['page']})" for r in results))

    return jsonify({"answer": answer, "sources": sources})

if __name__ == "__main__":
    app.run(debug=True, port=5000)