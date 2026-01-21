import re
import os
import pandas as pd
import numpy as np
import faiss
from flask import Flask, request, jsonify, render_template
from google import genai

# ---------------- FLASK APP ----------------

app = Flask(__name__)

# ---------------- CONFIGURATION ----------------

API_KEY = ""
client = genai.Client(api_key=API_KEY)

EMBEDDING_MODEL = "text-embedding-004"
LLM_MODEL = "gemini-2.5-flash"

CSV_PATH = "amazon.csv"
EMBEDDINGS_PATH = "review_embeddings.npy"
REVIEWS_PATH = "reviews.npy"

TOP_K = 10

# ---------------- STEP 1: PRODUCT ID EXTRACTION ----------------

def extract_product_id(url: str) -> str:
    patterns = [
        r"/dp/([A-Z0-9]{10})",
        r"/product/([a-zA-Z0-9]+)",
        r"pid=([a-zA-Z0-9]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Product ID not found in URL")

# ---------------- STEP 2: LOAD REVIEWS ----------------

def load_reviews(product_id: str):
    df = pd.read_csv(CSV_PATH)
    reviews = df[df["product_id"] == product_id]["review_content"].dropna().tolist()
    if not reviews:
        raise ValueError("No reviews found for this product")
    return reviews

# ---------------- STEP 3: EMBEDDINGS (ONE-TIME) ----------------

def generate_and_save_embeddings(reviews):
    vectors = []
    for review in reviews:
        res = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=review
        )
        vectors.append(res.embeddings[0].values)

    np.save(EMBEDDINGS_PATH, np.array(vectors, dtype="float32"))
    np.save(REVIEWS_PATH, np.array(reviews, dtype=object))

# ---------------- STEP 4: LOAD EMBEDDINGS ----------------

def load_embeddings():
    embeddings = np.load(EMBEDDINGS_PATH)
    reviews = np.load(REVIEWS_PATH, allow_pickle=True).tolist()
    return embeddings, reviews

# ---------------- STEP 5: FAISS ----------------

def build_faiss_index(embeddings):
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

# ---------------- STEP 6: RETRIEVE TOP-K ----------------

def retrieve_top_reviews(user_query, reviews, index):
    res = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=user_query
    )
    query_embedding = np.array([res.embeddings[0].values], dtype="float32")
    faiss.normalize_L2(query_embedding)
    _, indices = index.search(query_embedding, TOP_K)
    return [reviews[i] for i in indices[0]]

# ---------------- STEP 7: GENERATION ----------------

def generate_answer(user_query, retrieved_reviews):
    context = "\n".join(retrieved_reviews)

    prompt = f"""
You are an AI assistant that evaluates product suitability strictly based on user reviews.
Do NOT use external knowledge.

User Query:
{user_query}

Relevant Reviews:
{context}

Task:
Based ONLY on the reviews, state whether the product is worth buying.
Give a short justification.
"""

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )
    return response.text

# ---------------- ROUTES ----------------

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    print(">>> /analyze endpoint HIT")

    try:
        data = request.get_json()
        product_url = data.get("url")
        user_query = data.get("query")

        if not product_url or not user_query:
            return jsonify({"error": "Missing URL or query"}), 400

        product_id = extract_product_id(product_url)

        if not os.path.exists(EMBEDDINGS_PATH):
            reviews = load_reviews(product_id)
            generate_and_save_embeddings(reviews)

        embeddings, reviews = load_embeddings()
        index = build_faiss_index(embeddings)

        top_reviews = retrieve_top_reviews(user_query, reviews, index)
        answer = generate_answer(user_query, top_reviews)

        return jsonify({"answer": answer})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

# ---------------- RUN ----------------

if __name__ == "__main__":
    app.run(debug=True)
