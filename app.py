# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "sentence_transformers",
#   "faiss-cpu",
#   "requests",
#   "httpx",
#   "fastapi",
#   "uvicorn",
#   "slugify",
#   "pillow",
#   "requests",
#   "python-dotenv",
#   "google-genai",
# ]
# ///

import base64
import json
import mimetypes
import os
from io import BytesIO
from typing import Optional

import faiss
import httpx
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from google import genai
# Load env variables from .env
load_dotenv()

AIPROXY_URL = os.getenv("AIPROXY_URL")
# AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

if not AIPROXY_URL or not AIPROXY_TOKEN:
    raise RuntimeError("Missing AIPROXY_URL or AIPROXY_TOKEN in environment variables.")

# === Load resources ===
print("🔹 Loading FAISS index...")
index = faiss.read_index("faiss_combined_index.idx")

print("🔹 Loading metadata...")
with open("embedding_combined_data.json", "r", encoding="utf-8") as f:
    embedding_data = json.load(f)

print("🔹 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_image_mimetype(base64_string):
    image_data = base64.b64decode(base64_string)
    mime_type = None
    extension = None
    try:
        print("Image open")
        img = Image.open(BytesIO(image_data))
        img_type = img.format.lower()
        mime_type = f'image/{img_type}'
        extension = mimetypes.guess_extension(mime_type) or f".{img_type}"
    except Exception as e:
        print(f"Pillow detection failed: {e}")
    if not mime_type:
        print("in if not mime type")
        kind = imghdr.what(None, image_data)
        if kind:
            mime_type = f'image/{kind}'
            extension = mimetypes.guess_extension(mime_type) or f".{kind}"

    if not mime_type:
        raise ValueError("Could not determine the mimetype for your file — please ensure a valid image is provided.")

    return mime_type, extension, image_data


# Dummy placeholder for your Gemini integration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
def get_image_description(image_path):
    print("1")
    client = genai.Client(api_key=GEMINI_API_KEY)
    print("2")
    with open(image_path, "rb") as image_file:
        my_file = client.files.upload(file=image_path)  # or path=image_path
    print("3")

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            my_file,
            "Generate a description of image and also take care of highlighted parts if any"
        ]
    )
    print("4")
    return response.text

def query_faiss(query, top_k=5):
    query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    
    D, I = index.search(query_emb, top_k)
    
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < len(embedding_data):
            results.append({
                "score": float(score),
                **embedding_data[idx]
            })

    final_results = []
    for res in results:
        base_entry = {
            "score": res["score"],
            "text_snippet": (res.get("combined_text") or res.get("chunk") or "")[:200]
        }

        if "topic_title" in res:
            base_entry["topic_title"] = res["topic_title"]
        
        if "root_post_number" in res:
            base_entry["root_post_number"] = res["root_post_number"]
        
        # Handle discourse-style URLs (list of URLs)
        if "url" in res and isinstance(res["url"], list):
            for url in res["url"]:
                entry = base_entry.copy()
                entry["url"] = url
                final_results.append(entry)
        
        # Handle TDS / notes style
        elif "original_url" in res:
            entry = base_entry.copy()
            entry["url"] = res["original_url"]
            final_results.append(entry)
        
        else:
            # fallback case if no URL
            final_results.append(base_entry)

    return final_results

def generate_llm_response(query, context_texts):
    context = "\n\n---\n\n".join(context_texts)
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on forum discussions."
            },
            {
                "role": "user",
                "content": f"""<instructions>
You are a helpful assistant that answers questions based on forum discussions. 
First, think step-by-step about the relevant context. 
Provide your reasoning clearly, showing how each piece of the forum content contributes to your answer. 
Finally, give the answer in a concise and accurate way.
</instructions>

<context>
{context}
</context>

<question>
{query}
</question>

<format>
Respond in Markdown format:
- Use ### Reasoning as a section header for your reasoning
- Use ### Final Answer as a section header for your final answer
</format>
"""
            }
        ],
        "temperature": 0.3,
        "max_tokens": 400
    }

    response = httpx.post(AIPROXY_URL, headers=headers, json=payload, timeout=25.0)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        raise HTTPException(status_code=500, detail=f"AIPipe Error: {response.text}")



def answer(question, image):
    if image:
        print("got image")
        mime_type, extension, image_data = get_image_mimetype(image)
        filename = f"output_image{extension}"
        print("file_name")
        with open(filename, "wb") as f:
            f.write(image_data)
        print(f"Saved image to {filename}")
        image_response = get_image_description(filename)
        print("Got response")
        question = f"{question}\nImage description: {image_response}"

    results = query_faiss(question, top_k=10)
    # with open("faiss_results.json", "w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=2, ensure_ascii=False)

    print("FAISS results saved to faiss_results.json")
    url_text_list = []
    for item in results:
        url = item.get("url", "")
        text = item.get("text_snippet", "")
        url_text_list.append({
            "url": url,
            "text": text
        })
    context_texts = []
    for entry in url_text_list:
        url = entry.get("url", "")
        text = entry.get("text", "")
        combined = f"{url} {text}"
        context_texts.append(combined)
  
    # Get answer text from model
    response_text = generate_llm_response(question, context_texts)
    response_text = response_text or "No answer generated."

    # Save to a file
    # with open("llm_response.txt", "w", encoding="utf-8") as f:
    #     f.write(response_text)

    print("LLM response saved to llm_response.txt")

    return JSONResponse({
        "answer": response_text,
        "links": url_text_list
    })


# === FastAPI app ===
app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "FastAPI is running!"}

class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

@app.post("/api")
def api_answer(request: QueryRequest):
    try:
        return answer(request.question, request.image)
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=7860)
