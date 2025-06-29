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
from serpapi import GoogleSearch  # 🆕 New import

# --- Load environment variables ---
load_dotenv()
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")  # 🆕 Added
INDEX_NAME = "chatbot-index"
DATA_FOLDER = "data"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Gemini Setup ---
genai.configure(api_key=GENAI_API_KEY)

# --- Tokenizer ---
encoding = tiktoken.get_encoding("cl100k_base")

# --- Session Memory ---
user_sessions = {}

# --- Embedding ---
def get_embedding(text):
    res = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return res["embedding"] if isinstance(res, dict) else res.embedding

# --- Pinecone Setup ---
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

# --- SerpAPI Fallback ---
def search_google(query):
    try:
        params = {
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "engine": "google",
            "num": 3,
            "hl": "en"
        }
        search = GoogleSearch(params)
        results = search.get_dict()

        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]

        snippets = [r.get("snippet") for r in results.get("organic_results", []) if "snippet" in r]
        if snippets:
            return "\n\n".join(snippets)

        return "I couldn't find any relevant results on Google."
    except Exception as e:
        logger.error(f"Google Search error: {e}")
        return "Sorry, I couldn't fetch results from Google."

# --- Chat Logic ---
def chat_with_bot(query, memory_context=""):
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=10, include_metadata=True)
    matches = [m for m in results["matches"] if m["score"] > 0.5]
    document_context = "\n".join([m["metadata"]["text"] for m in matches])

    if "executive summary" in query.lower():
        prompt = f"""
You are a great data expert. You need to create an executive summary of around 700–1000 words from the following content of a document.

The content is:
{document_context}


 Provide the output in a professional and concise tone, suitable for a business executive. Avoid phrases like \"this document says\" or \"based on the context\". Just deliver the  executive summary confidently.

 Follow these formatting rules:
 - Use headings or sub-headings only when they help clarify.
 - Restrictly don't use ( * , ** ) charaters just bold that sentences.
 - If bullet points are required, place each on a new line (no special characters or numbering).
 - don't use points like recommendations ,conslusions, challenges kinds of points. just give executive summary in a clean, concise, and professional tone using plain language  

"""
    elif "elevator pitch" in query.lower():
        prompt = f"""
You are a business analyst. Create an elevator pitch in 3 styles based on the content below.
Content:
{document_context}

 Provide the following two things as output in  format:
 1) The 'elevator pitch' of 4–5 sentences in three different styles: 
 2) Make sure each pitch is clear, impactful, and suitable for quick communication.
 Follow these formatting rules:
 - Use headings or sub-headings only when they help clarify.
 - Restrictly don't use ( * , ** ) charaters just bold that sentences.
 - If bullet points are required, place each on a new line (no special characters or numbering).
"""
    elif "what can you tell me" in query.lower() or "how can you assist me" in query.lower():
        prompt = f"""
You are an AI assistant built by WLGA company.

If a user asks about your role, purpose, or how you can help — 
such as questions like "What can you do?" or "How can you assist me?" — 
you should reply with:

"I can assist you to know about WLGA company."

Now, answer the user's query below:
User: {query}
"""
    else:
        prompt = f"""
Answer the user's question in a clear, helpful tone using the information below. If not found, use your own knowledge.
 You are a knowledgeable and helpful AI assistant.

 You can use the information below to answer. If the exact answer is not found, respond using your own knowledge — but never mention whether the information came from a document or memory. Just provide the best possible answer confidently and naturally.

 Follow these formatting rules:
 - Keep paragraphs short and break them for readability when needed.
 - If the user asks about your knowledge, capabilities, or how many documents/sources you contain, respond naturally. Do not use phrases like \"I do not\", \"I cannot\", \"I don’t have access\", or mention limitations.
 - You have to understand the pattern of language and then respond.
 - Just give a general, helpful, confident answer — assuming you are capable of answering from a wide range of relevant knowledge.
 - Use headings or sub-headings only when they help clarify.
 - Restrictly don't use ( * , ** ) charaters just bold that sentences.
 - If bullet points are required, place each on a new line (no special characters or numbering).
 - Never include phrases like \"the document says\", \"based on the context\", \"I do not know\", \"as an AI\", or any reference to documents, memory, or limitations.

Chat history:
{memory_context}

Reference:
{document_context}

User's question:
{query}

Answer:
"""


    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        if not response.text.strip():
            logger.info("No response from Gemini. Trying Google Search...")
            return search_google(query)

        if len(matches) == 0:
            logger.info("No relevant PDF matches found. Falling back to Google...")
            return search_google(query)

        return response.text

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return search_google(query)  # 🆕 Google fallback on error

# --- Intent Detection ---
def detect_intent(user_query):
    clear_phrases = [
        "clear", "reset", "delete", "start over", "new chat", "forget everything",
        "clear chat", "clear conversation", "reset history", "delete all","refresh chat","delete history","refresh"
    ]
    user_query = user_query.lower()
    return "clear_chat" if any(phrase in user_query for phrase in clear_phrases) else "chat"

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

    if detect_intent(query) == "clear_chat":
        user_sessions.pop(session_id, None)
        return jsonify({"response": "", "session_id": str(uuid.uuid4())})

    if session_id not in user_sessions:
        user_sessions[session_id] = deque(maxlen=5)

    user_sessions[session_id].append({"role": "user", "text": query})
    memory_context = "\n".join([f"{m['role'].capitalize()}: {m['text']}" for m in user_sessions[session_id]])

    answer = chat_with_bot(query, memory_context)
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
# from collections import deque
# import uuid


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

# # --- Session-based memory store ---
# user_sessions = {}  # session_id: deque of messages

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
# def chat_with_bot(query, memory_context=""):
#     query_embedding = get_embedding(query)
#     results = index.query(
#         vector=query_embedding,
#         top_k=10,
#         include_metadata=True
#     )
#     relevant_matches = [m for m in results["matches"] if m["score"] > 0.5]
#     document_context = "\n".join([match["metadata"]["text"] for match in relevant_matches])

#     # Handle vague/general queries
    # if any(phrase in query.lower() for phrase in ["what can you tell","what can you tell me", "can you help me", "what do you do", "how can you assist me ", "what is your purpose","what is your role", "what can you do", "how can you help me", "what is your function", "what is your job", "what is your task","what is your capability", "what is your expertise", "what is your knowledge", "what is your skill", "what is your ability","what is your service", "what is your offering", "what is your feature", "what is your function", "what is your role in this context","what is your role in this conversation", "what is your role in this interaction","how can you help me ", "how can you assist me", "how can you support me", "how can you guide me", "how can you provide information"]):
    #  return "I can assist you to know about  WLGA company."


#     # --- Executive Summary Condition ---
#     if "executive summary" in query.lower():
#         prompt = f"""
# You are a great data expert. You need to create an executive summary of around 700–1000 words from the following content of a document.

# The content is:
# {document_context}


# Provide the output in a professional and concise tone, suitable for a business executive. Avoid phrases like \"this document says\" or \"based on the context\". Just deliver the  executive summary confidently.

# Follow these formatting rules:
# - Use headings or sub-headings only when they help clarify.
# - If bullet points are required, place each on a new line (no special characters or numbering).
# - don't use points like recommendations ,conslusions, challenges kinds of points. just give executive summary in a clean, concise, and professional tone using plain language  

# Your response:
# """
#     elif "elevator pitch" in query.lower():
#         prompt = f"""
# You are a great business analyst. You need to create an elevator pitch based on the following content from a document.

# The content is:
# {document_context}

# Provide the following two things as output in  format:
# 1) The 'elevator pitch' of 4–5 sentences in three different styles: 
# 2) Make sure each pitch is clear, impactful, and suitable for quick communication.
# Follow these formatting rules:
# - Use headings or sub-headings only when they help clarify.
# - If bullet points are required, place each on a new line (no special characters or numbering).
# """
#     else:
#         prompt = f"""
# You are a knowledgeable and helpful AI assistant.

# Answer the following question in a clean, concise, and professional tone using plain language. Your response should be relevant, easy to understand, and structured in a user-friendly format.

# You can use the information below to answer. If the exact answer is not found, respond using your own knowledge — but never mention whether the information came from a document or memory. Just provide the best possible answer confidently and naturally.

# Follow these formatting rules:
# - Keep paragraphs short and break them for readability when needed.
# - If the user asks about your knowledge, capabilities, or how many documents/sources you contain, respond naturally. Do not use phrases like \"I do not\", \"I cannot\", \"I don’t have access\", or mention limitations.
# - You have to understand the pattern of language and then respond.
# - Just give a general, helpful, confident answer — assuming you are capable of answering from a wide range of relevant knowledge.
# - Use headings or sub-headings only when they help clarify.
# - If bullet points are required, place each on a new line (no special characters or numbering).
# - Never include phrases like \"the document says\", \"based on the context\", \"I do not know\", \"as an AI\", or any reference to documents, memory, or limitations.

# Recent conversation history:
# {memory_context}

# Reference information:
# {document_context}

# User's question:
# {query}

# Answer:
# """

#     try:
#         model = genai.GenerativeModel("gemini-1.5-flash")
#         response = model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         logger.error(f"Chat error: {e}")
#         return "I encountered an error processing your request. Please try again."


# def detect_intent(user_query):
#     clear_phrases = [
#         "clear chat", "reset", "delete conversation", "clear history",
#         "start over", "new chat", "forget everything","clear", "clear chat", "reset", "delete conversation", "clear conversation", "reset chat", "delete chat", "clear history", "reset history","delete history","clear chat history", "reset chat history", "delete chat history", "clear conversation history", "reset conversation history", "delete conversation history","clear all", "reset all", "delete all","clear all history", "reset all history", "delete all history","clear the chat", "reset the chat", "delete the chat","clear this chat", "reset this chat", "delete this chat","clear this conversation", "reset this conversation", "delete this conversation","clear the conversation", "reset the conversation", "delete the conversation","clear the history","reset the history", "delete the history","clear everything", "reset everything", "delete everything","clear all conversations", "reset all conversations", "delete all conversations","clear all chats", "reset all chats", "delete all chats","clear this chat history", "reset this chat history", "delete this chat history","clear this conversation history", "reset this conversation history", "delete this conversation history","refresh the chat","refresh the page","refresh this page","refresh this chat","clear this page"
#     ]
#     user_query = user_query.lower().strip()
    
#     if any(phrase in user_query for phrase in clear_phrases):
#         return "clear_chat"
#     return "chat"

# # --- Flask Setup ---
# app = Flask(__name__)
# CORS(app)

# @app.route("/chat", methods=["POST"])
# def chat_api():
#     data = request.get_json()
#     query = data.get("query", "").strip()
#     session_id = data.get("session_id", str(uuid.uuid4()))
   

#     if not query:
#         return jsonify({"error": "No query provided"}), 400

#     # Handle intent to clear chat
#     if detect_intent(query) == "clear_chat":
#         user_sessions.pop(session_id, None)
#         return jsonify({"response": "", "session_id": str(uuid.uuid4())})

#     # Initialize session if needed
#     if session_id not in user_sessions:
#         user_sessions[session_id] = deque(maxlen=5)

#     # Add user message
#     user_sessions[session_id].append({"role": "user", "text": query})

#     # Build memory context
#     memory_context = ""
#     for msg in user_sessions[session_id]:
#         memory_context += f"{msg['role'].capitalize()}: {msg['text']}\n"
 


#     # Get bot answer
#     answer = chat_with_bot(query, memory_context)
#     # Store bot reply
#     user_sessions[session_id].append({"role": "bot", "text": answer})

#     return jsonify({"response": answer, "session_id": session_id})

# # --- Start Server ---
# if __name__ == "__main__":
#     index_documents()
#     app.run(debug=False, host="0.0.0.0", port=5000)







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
# from collections import deque
# import uuid

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

# # --- Session-based memory store ---
# user_sessions = {}  # session_id: deque of messages

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
# def chat_with_bot(query, memory_context=""):
#     query_embedding = get_embedding(query)
#     results = index.query(
#         vector=query_embedding,
#         top_k=10,
#         include_metadata=True
#     )
#     relevant_matches = [m for m in results["matches"] if m["score"] > 0.5]
#     document_context = "\n".join([match["metadata"]["text"] for match in relevant_matches])

#     # Handle vague/general queries
#     if any(phrase in query.lower() for phrase in ["what can you tell","what can you tell me", "can you help me", "what do you do", "how can you assist me ", "what is your purpose","what is your role", "what can you do", "how can you help me", "what is your function", "what is your job", "what is your task","what is your capability", "what is your expertise", "what is your knowledge", "what is your skill", "what is your ability","what is your service", "what is your offering", "what is your feature", "what is your function", "what is your role in this context","what is your role in this conversation", "what is your role in this interaction","how can you help me ", "how can you assist me", "how can you support me", "how can you guide me", "how can you provide information"]):
#      return "I can assist you to know about  WLGA company."


#     # --- Executive Summary Condition ---
#     if "executive summary" in query.lower():
#         prompt = f"""
# You are a great data expert. You need to create an executive summary of around 700–1000 words from the following content of a document.

# The content is:
# {document_context}


# Provide the output in a professional and concise tone, suitable for a business executive. Avoid phrases like \"this document says\" or \"based on the context\". Just deliver the  executive summary confidently.

# Follow these formatting rules:
# - Use headings or sub-headings only when they help clarify.
# - If bullet points are required, place each on a new line (no special characters or numbering).
# - don't use points like recommendations ,conslusions, challenges kinds of points. just give executive summary in a clean, concise, and professional tone using plain language  

# Your response:
# """
#     elif "elevator pitch" in query.lower():
#         prompt = f"""
# You are a great business analyst. You need to create an elevator pitch based on the following content from a document.

# The content is:
# {document_context}

# Provide the following two things as output in  format:
# 1) The 'elevator pitch' of 4–5 sentences in three different styles: 
# 2) Make sure each pitch is clear, impactful, and suitable for quick communication.
# Follow these formatting rules:
# - Use headings or sub-headings only when they help clarify.
# - If bullet points are required, place each on a new line (no special characters or numbering).
# """
#     else:
#         prompt = f"""
# You are a knowledgeable and helpful AI assistant.

# Answer the following question in a clean, concise, and professional tone using plain language. Your response should be relevant, easy to understand, and structured in a user-friendly format.

# You can use the information below to answer. If the exact answer is not found, respond using your own knowledge — but never mention whether the information came from a document or memory. Just provide the best possible answer confidently and naturally.

# Follow these formatting rules:
# - Keep paragraphs short and break them for readability when needed.
# - If the user asks about your knowledge, capabilities, or how many documents/sources you contain, respond naturally. Do not use phrases like \"I do not\", \"I cannot\", \"I don’t have access\", or mention limitations.
# - You have to understand the pattern of language and then respond.
# - Just give a general, helpful, confident answer — assuming you are capable of answering from a wide range of relevant knowledge.
# - Use headings or sub-headings only when they help clarify.
# - If bullet points are required, place each on a new line (no special characters or numbering).
# - Never include phrases like \"the document says\", \"based on the context\", \"I do not know\", \"as an AI\", or any reference to documents, memory, or limitations.

# Recent conversation history:
# {memory_context}

# Reference information:
# {document_context}

# User's question:
# {query}

# Answer:
# """
# import json

# def detect_intent(user_query):
#     prompt = f"""
# You're an intent detection system.

# Read the user's message and identify their intent.

# Respond with a single word like:
# - "clear_chat" if they want to reset or clear everything
# - "chat" if it's just a normal question

# Message:
# "{user_query}"

# Only respond in JSON: {{ "intent": "..." }}
# """
#     try:
#         response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
#         print("Intent raw response:", response.text)
#         return json.loads(response.text.strip()).get("intent", "").lower()
#     except Exception as e:
#         print("Intent detection failed:", e)
#         return "chat"




#     # response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
#     # return response.text.strip()





#     # # if query.lower() in ["clear", "clear chat", "reset", "delete conversation", "clear conversation", "reset chat", "delete chat", "clear history", "reset history","delete history","clear chat history", "reset chat history", "delete chat history", "clear conversation history", "reset conversation history", "delete conversation history","clear all", "reset all", "delete all","clear all history", "reset all history", "delete all history","clear the chat", "reset the chat", "delete the chat","clear this chat", "reset this chat", "delete this chat","clear this conversation", "reset this conversation", "delete this conversation","clear the conversation", "reset the conversation", "delete the conversation"]:
#     # #  user_sessions[session_id].clear()
#     # #  return jsonify({
#     # #     "response": "Chat history has been cleared.",
#     # #     "session_id": session_id
#     # # })


# # --- Flask Setup ---
# app = Flask(__name__)
# CORS(app)

# @app.route("/chat", methods=["POST"])
# def chat_api():
#     data = request.get_json()
#     query = data.get("query", "").strip()
#     session_id = data.get("session_id", str(uuid.uuid4()))
#     print("User query:", query)

#     if not query:
#         return jsonify({"error": "No query provided"}), 400

#     # Handle intent to clear chat
#     if detect_intent(query) == "clear_chat":
#         user_sessions.pop(session_id, None)
#         return jsonify({"response": "", "session_id": str(uuid.uuid4())})

#     # Initialize session if needed
#     if session_id not in user_sessions:
#         user_sessions[session_id] = deque(maxlen=5)

#     # Add user message
#     user_sessions[session_id].append({"role": "user", "text": query})

#     # Build memory context
#     memory_context = ""
#     for msg in user_sessions[session_id]:
#         memory_context += f"{msg['role'].capitalize()}: {msg['text']}\n"
#     print("Memory context:", memory_context)  # ✅ ADD THIS LINE


#     # Get bot answer
#     answer = chat_with_bot(query, memory_context)
#     print("Bot answer:", answer)  # ✅ ADD THIS LINE
#     # Store bot reply
#     user_sessions[session_id].append({"role": "bot", "text": answer})

#     return jsonify({"response": answer, "session_id": session_id})

# # --- Start Server ---
# if __name__ == "__main__":
#     index_documents()
#     app.run(debug=False, host="0.0.0.0", port=5000)







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
# from collections import deque
# import uuid

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

# # --- Session-based memory store ---
# user_sessions = {}  # session_id: deque of messages

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
# def chat_with_bot(query, memory_context=""):
#     query_embedding = get_embedding(query)
#     results = index.query(
#         vector=query_embedding,
#         top_k=10,
#         include_metadata=True
#     )
#     relevant_matches = [m for m in results["matches"] if m["score"] > 0.5]
#     document_context = "\n".join([match["metadata"]["text"] for match in relevant_matches])

#     # Handle vague/general queries
#     if any(phrase in query.lower() for phrase in ["what can you tell","what can you tell me", "can you help me", "what do you do", "how can you assist me ", "what is your purpose","what is your role", "what can you do", "how can you help me", "what is your function", "what is your job", "what is your task","what is your capability", "what is your expertise", "what is your knowledge", "what is your skill", "what is your ability","what is your service", "what is your offering", "what is your feature", "what is your function", "what is your role in this context","what is your role in this conversation", "what is your role in this interaction","how can you help me ", "how can you assist me", "how can you support me", "how can you guide me", "how can you provide information"]):
#      return "I can assist you to know about  WLGA company."


#     # --- Executive Summary Condition ---
#     if "executive summary" in query.lower():
#         prompt = f"""
# You are a great data expert. You need to create an executive summary of around 700–1000 words from the following content of a document.

# The content is:
# {document_context}


# Provide the output in a professional and concise tone, suitable for a business executive. Avoid phrases like \"this document says\" or \"based on the context\". Just deliver the  executive summary confidently.

# Follow these formatting rules:
# - Use headings or sub-headings only when they help clarify.
# - If bullet points are required, place each on a new line (no special characters or numbering).
# - don't use points like recommendations ,conslusions, challenges kinds of points. just give executive summary in a clean, concise, and professional tone using plain language  

# Your response:
# """
#     elif "elevator pitch" in query.lower():
#         prompt = f"""
# You are a great business analyst. You need to create an elevator pitch based on the following content from a document.

# The content is:
# {document_context}

# Provide the following two things as output in  format:
# 1) The 'elevator pitch' of 4–5 sentences in three different styles: 
# 2) Make sure each pitch is clear, impactful, and suitable for quick communication.
# Follow these formatting rules:
# - Use headings or sub-headings only when they help clarify.
# - If bullet points are required, place each on a new line (no special characters or numbering).
# """
#     else:
#         prompt = f"""
# You are a knowledgeable and helpful AI assistant.

# Answer the following question in a clean, concise, and professional tone using plain language. Your response should be relevant, easy to understand, and structured in a user-friendly format.

# You can use the information below to answer. If the exact answer is not found, respond using your own knowledge — but never mention whether the information came from a document or memory. Just provide the best possible answer confidently and naturally.

# Follow these formatting rules:
# - Keep paragraphs short and break them for readability when needed.
# - If the user asks about your knowledge, capabilities, or how many documents/sources you contain, respond naturally. Do not use phrases like \"I do not\", \"I cannot\", \"I don’t have access\", or mention limitations.
# - You have to understand the pattern of language and then respond.
# - Just give a general, helpful, confident answer — assuming you are capable of answering from a wide range of relevant knowledge.
# - Use headings or sub-headings only when they help clarify.
# - If bullet points are required, place each on a new line (no special characters or numbering).
# - Never include phrases like \"the document says\", \"based on the context\", \"I do not know\", \"as an AI\", or any reference to documents, memory, or limitations.

# Recent conversation history:
# {memory_context}

# Reference information:
# {document_context}

# User's question:
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
#     session_id = data.get("session_id", str(uuid.uuid4()))

#     if not query:
#         return jsonify({"error": "No query provided"}), 400
    

#     if query.lower() in ["clear", "clear chat", "reset", "delete conversation", "clear conversation", "reset chat", "delete chat", "clear history", "reset history","delete history","clear chat history", "reset chat history", "delete chat history", "clear conversation history", "reset conversation history", "delete conversation history","clear all", "reset all", "delete all","clear all history", "reset all history", "delete all history","clear the chat", "reset the chat", "delete the chat","clear this chat", "reset this chat", "delete this chat","clear this conversation", "reset this conversation", "delete this conversation","clear the conversation", "reset the conversation", "delete the conversation"]:
#      user_sessions[session_id].clear()
#      return jsonify({
#         "response": "Chat history has been cleared.",
#         "session_id": session_id
#     })

#     # Initialize session if needed
#     if session_id not in user_sessions:
#         user_sessions[session_id] = deque(maxlen=5)

#     # Add user message
#     user_sessions[session_id].append({"role": "user", "text": query})

#     # Build memory context
#     memory_context = ""
#     for msg in user_sessions[session_id]:
#         memory_context += f"{msg['role'].capitalize()}: {msg['text']}\n"

#     # Get bot answer
#     answer = chat_with_bot(query, memory_context)

#     # Store bot reply
#     user_sessions[session_id].append({"role": "bot", "text": answer})

#     return jsonify({"response": answer, "session_id": session_id})

# # --- Start Server ---
# if __name__ == "__main__":
#     index_documents()
#     app.run(debug=False, host="0.0.0.0", port=5000)




























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
# from collections import deque
# import uuid

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

# # --- Session-based memory store ---
# user_sessions = {}  # session_id: deque of messages

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
# def chat_with_bot(query, memory_context=""):
#     query_embedding = get_embedding(query)
#     results = index.query(
#         vector=query_embedding,
#         top_k=10,
#         include_metadata=True
#     )
#     relevant_matches = [m for m in results["matches"] if m["score"] > 0.5]

#     document_context = "\n".join([match["metadata"]["text"] for match in relevant_matches])

#     prompt = f"""
# You are a knowledgeable and helpful AI assistant.

# Answer the following question in a clean, concise, and professional tone using plain language. Your response should be relevant, easy to understand, and structured in a user-friendly format.

# You can use the information below to answer. If the exact answer is not found, respond using your own knowledge — but never mention whether the information came from a document or memory. Just provide the best possible answer confidently and naturally.

# Follow these formatting rules:
# - Keep paragraphs short and break them for readability when needed.
# - If the user asks about your knowledge, capabilities, or how many documents/sources you contain, respond naturally. Do not use phrases like "I do not", "I cannot", "I don’t have access", or mention limitations.
# -you have to understand the pattern of language and then respond.
# - Just give a general, helpful, confident answer — assuming you are capable of answering from a wide range of relevant knowledge.
# - Use headings or sub-headings only when they help clarify.
# - If bullet points are required, place each on a new line (no special characters or numbering).
# - Never include phrases like "the document says", "based on the context", "I do not know", "as an AI", or any reference to documents, memory, or limitations.

# Recent conversation history:
# {memory_context}

# Reference information:
# {document_context}

# User's question:
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
#     session_id = data.get("session_id", str(uuid.uuid4()))

#     if not query:
#         return jsonify({"error": "No query provided"}), 400

#     # Initialize session if needed
#     if session_id not in user_sessions:
#         user_sessions[session_id] = deque(maxlen=5)

#     # Add user message
#     user_sessions[session_id].append({"role": "user", "text": query})

#     # Build memory context
#     memory_context = ""
#     for msg in user_sessions[session_id]:
#         memory_context += f"{msg['role'].capitalize()}: {msg['text']}\n"

#     # Get bot answer
#     answer = chat_with_bot(query, memory_context)

#     # Store bot reply
#     user_sessions[session_id].append({"role": "bot", "text": answer})

#     return jsonify({"response": answer, "session_id": session_id})

# # --- Start Server ---
# if __name__ == "__main__":
#     index_documents()
#     app.run(debug=False, host="0.0.0.0", port=5000)


















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



