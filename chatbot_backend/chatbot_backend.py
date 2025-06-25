import os
import fitz  # PyMuPDF
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
import tiktoken
import logging
from dotenv import load_dotenv
from collections import deque
import uuid

# --- Load environment variables ---
load_dotenv()
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
INDEX_NAME = "chatbot-index"
DATA_FOLDER = "data"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Gemini Setup ---
genai.configure(api_key=GENAI_API_KEY)

# --- Tokenizer ---
encoding = tiktoken.get_encoding("cl100k_base")

# --- Session-based memory store ---
user_sessions = {}  # session_id: deque of messages

# --- Embedding ---
def get_embedding(text):
    res = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return res["embedding"] if isinstance(res, dict) else res.embedding

# --- Init Pinecone ---
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    logger.info("Creating Pinecone index...")
    TEST_EMBED_DIM = len(get_embedding("test"))
    pc.create_index(
        name=INDEX_NAME,
        dimension=TEST_EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
index = pc.Index(INDEX_NAME)

# --- PDF Reading ---
def read_pdf(filepath):
    text = ""
    try:
        doc = fitz.open(filepath)
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            image_blocks = [b for b in page.get_text("dict")["blocks"] if b["type"] == 1]
            for _ in image_blocks:
                text += f"\n[Image on page {page_num+1}: Possibly a graph or chart. Provide context if relevant.]\n"
            text += page_text + "\n"
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
    return text

# --- Token-based Chunking ---
def split_text_by_tokens(text, max_tokens=400):
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text.strip())
    return chunks

# --- Indexing ---
def index_documents():
    logger.info("Indexing documents...")
    all_chunks = []

    for filename in os.listdir(DATA_FOLDER):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(DATA_FOLDER, filename)
            if os.path.isfile(path):
                logger.info(f"Reading: {filename}")
                content = read_pdf(path)
                chunks = split_text_by_tokens(content)
                for i, chunk in enumerate(chunks):
                    if chunk:
                        all_chunks.append((f"{filename}-{i}", chunk, filename))

    logger.info(f"Total chunks to index: {len(all_chunks)}")
    vectors = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_id = {
            executor.submit(get_embedding, chunk): (doc_id, chunk, fname)
            for doc_id, chunk, fname in all_chunks
        }
        for future in as_completed(future_to_id):
            doc_id, chunk, fname = future_to_id[future]
            try:
                embedding = future.result()
                vectors.append({
                    "id": doc_id,
                    "values": embedding,
                    "metadata": {"text": chunk, "source": fname}
                })
            except Exception as e:
                logger.error(f"Embedding failed for {doc_id}: {e}")

    logger.info("Uploading to Pinecone...")
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)
        logger.info(f"Uploaded batch {i // batch_size + 1}")

    logger.info("Indexing complete.")

# --- Chat Logic ---
def chat_with_bot(query, memory_context=""):
    query_embedding = get_embedding(query)
    results = index.query(
        vector=query_embedding,
        top_k=10,
        include_metadata=True
    )
    relevant_matches = [m for m in results["matches"] if m["score"] > 0.5]

    document_context = "\n".join([match["metadata"]["text"] for match in relevant_matches])

    prompt = f"""
You are a knowledgeable AI assistant.

Answer the following question in a clean and professional tone using plain language. The output should be very relevant, concise, and easy to understand.

Follow these rules for formatting:
- Break longer paragraphs into shorter lines where it improves readability.
- Add a heading or sub-heading only if it improves clarity.
- If bullet points are needed, start each one on a new line (no '*', '-', or numbering).
- Keep the tone informative, clear, and natural like a human expert.

Recent conversation:
{memory_context}

Document context:
{document_context}

User's current question:
{query}

Answer:
"""
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
    return response.text.strip()

# --- Flask Setup ---
app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat_api():
    data = request.get_json()
    query = data.get("query", "").strip()
    session_id = data.get("session_id", str(uuid.uuid4()))

    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Initialize session if needed
    if session_id not in user_sessions:
        user_sessions[session_id] = deque(maxlen=5)

    # Add user message
    user_sessions[session_id].append({"role": "user", "text": query})

    # Build memory context
    memory_context = ""
    for msg in user_sessions[session_id]:
        memory_context += f"{msg['role'].capitalize()}: {msg['text']}\n"

    # Get bot answer
    answer = chat_with_bot(query, memory_context)

    # Store bot reply
    user_sessions[session_id].append({"role": "bot", "text": answer})

    return jsonify({"response": answer, "session_id": session_id})

# --- Start Server ---
if __name__ == "__main__":
    index_documents()
    app.run(debug=False, host="0.0.0.0", port=5000)






# import os
# import fitz  # PyMuPDF
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from pinecone import Pinecone, ServerlessSpec
# import google.generativeai as genai
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import tiktoken
# import logging
# from dotenv import load_dotenv

# # --- Load environment variables ---
# load_dotenv()
# GENAI_API_KEY = os.getenv("GENAI_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
# INDEX_NAME = "chatbot-index"
# DATA_FOLDER = "data"

# # --- Logging Setup ---
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # --- Gemini Setup ---
# genai.configure(api_key=GENAI_API_KEY)

# # --- Tokenizer ---
# encoding = tiktoken.get_encoding("cl100k_base")

# # --- Embedding ---
# def get_embedding(text):
#     res = genai.embed_content(
#         model="models/embedding-001",
#         content=text,
#         task_type="retrieval_document"
#     )
#     return res["embedding"] if isinstance(res, dict) else res.embedding

# # --- Init Pinecone ---
# pc = Pinecone(api_key=PINECONE_API_KEY)
# if INDEX_NAME not in pc.list_indexes().names():
#     logger.info("Creating Pinecone index...")
#     TEST_EMBED_DIM = len(get_embedding("test"))
#     pc.create_index(
#         name=INDEX_NAME,
#         dimension=TEST_EMBED_DIM,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
#     )
# index = pc.Index(INDEX_NAME)

# # --- PDF Reading ---
# def read_pdf(filepath):
#     text = ""
#     try:
#         doc = fitz.open(filepath)
#         for page_num, page in enumerate(doc):
#             page_text = page.get_text()
#             image_blocks = [b for b in page.get_text("dict")["blocks"] if b["type"] == 1]
#             for _ in image_blocks:
#                 text += f"\n[Image on page {page_num+1}: Possibly a graph or chart. Provide context if relevant.]\n"
#             text += page_text + "\n"
#     except Exception as e:
#         logger.error(f"Error reading {filepath}: {e}")
#     return text

# # --- Token-based Chunking ---
# def split_text_by_tokens(text, max_tokens=400):
#     tokens = encoding.encode(text)
#     chunks = []
#     for i in range(0, len(tokens), max_tokens):
#         chunk_tokens = tokens[i:i+max_tokens]
#         chunk_text = encoding.decode(chunk_tokens)
#         chunks.append(chunk_text.strip())
#     return chunks

# # --- Indexing ---
# def index_documents():
#     logger.info("Indexing documents...")
#     all_chunks = []

#     for filename in os.listdir(DATA_FOLDER):
#         if filename.lower().endswith(".pdf"):
#             path = os.path.join(DATA_FOLDER, filename)
#             if os.path.isfile(path):
#                 logger.info(f"Reading: {filename}")
#                 content = read_pdf(path)
#                 chunks = split_text_by_tokens(content)
#                 for i, chunk in enumerate(chunks):
#                     if chunk:
#                         all_chunks.append((f"{filename}-{i}", chunk, filename))

#     logger.info(f"Total chunks to index: {len(all_chunks)}")
#     vectors = []
#     with ThreadPoolExecutor(max_workers=8) as executor:
#         future_to_id = {
#             executor.submit(get_embedding, chunk): (doc_id, chunk, fname)
#             for doc_id, chunk, fname in all_chunks
#         }
#         for future in as_completed(future_to_id):
#             doc_id, chunk, fname = future_to_id[future]
#             try:
#                 embedding = future.result()
#                 vectors.append({
#                     "id": doc_id,
#                     "values": embedding,
#                     "metadata": {"text": chunk, "source": fname}
#                 })
#             except Exception as e:
#                 logger.error(f"Embedding failed for {doc_id}: {e}")

#     logger.info("Uploading to Pinecone...")
#     batch_size = 100
#     for i in range(0, len(vectors), batch_size):
#         batch = vectors[i:i+batch_size]
#         index.upsert(vectors=batch)
#         logger.info(f"Uploaded batch {i // batch_size + 1}")

#     logger.info("Indexing complete.")

# # --- Chat Logic ---
# def chat_with_bot(query):
#     query_embedding = get_embedding(query)
#     results = index.query(
#         vector=query_embedding,
#         top_k=10,
#         include_metadata=True
#     )
#     relevant_matches = [m for m in results["matches"] if m["score"] > 0.5]
#     if not relevant_matches:
#         return "Sorry, I couldn't find anything relevant in the documents."

#     context = "\n".join([match["metadata"]["text"] for match in relevant_matches])
#     prompt = f"""
# You are a knowledgeable AI assistant.

# Answer the following question in a clean and professional tone using plain language. The output should be very relevant, concise, and easy to understand.

# Follow these rules for formatting:
# - Break longer paragraphs into shorter lines where it improves readability.
# - Add a heading or sub-heading only if it improves clarity.
# - If bullet points are needed, start each one on a new line (no '*', '-', or numbering).
# - Keep the tone informative, clear, and natural like a human expert.

# Context:
# {context}

# Question:
# {query}

# Answer:
# """
#     response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
#     return response.text.strip()

# # --- Flask Setup ---
# app = Flask(__name__)
# CORS(app)

# @app.route("/chat", methods=["POST"])
# def chat_api():
#     data = request.get_json()
#     query = data.get("query", "").strip()
#     if not query:
#         return jsonify({"error": "No query provided"}), 400

#     answer = chat_with_bot(query)
#     return jsonify({"response": answer})

# # --- Start Server ---
# if __name__ == "__main__":
#     index_documents()
#     app.run(debug=False, host="0.0.0.0", port=5000)



