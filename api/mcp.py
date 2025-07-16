from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import os
from openai import OpenAI  # ✅ New SDK client
from pinecone import Pinecone

app = FastAPI()

# ✅ CORSMiddleware for your Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔒 Replace with your real frontend domain
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ✅ Initialize OpenAI & Pinecone clients
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

# ✅ Request body model
class QueryRequest(BaseModel):
    query: str
    filters: Optional[Dict] = None
    top_k: Optional[int] = 5

# ✅ POST endpoint for /mcp
@app.post("/mcp")
async def mcp_search(payload: QueryRequest, request: Request):
    try:
        # ✅ New SDK method to generate embeddings
        response = client.embeddings.create(
            input=payload.query,
            model="text-embedding-3-small"
        )
        embed = response.data[0].embedding

        # ✅ Query Pinecone
        result = index.query(
            vector=embed,
            top_k=payload.top_k,
            include_metadata=True,
            filter=payload.filters or {}
        )

        return JSONResponse(content={"results": result["matches"]})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
