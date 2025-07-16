from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import openai
import os
from pinecone import Pinecone

app = FastAPI()

# ✅ CORSMiddleware for your Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://vercel.com"],  # ✅ Replace with your actual deployed frontend URL
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],  # Use wildcard if you're not sure what frontend sends
)

# ✅ Environment variables
openai.api_key = os.environ["OPENAI_API_KEY"]
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
        embed = openai.Embedding.create(
            input=payload.query,
            model="text-embedding-3-small"
        )["data"][0]["embedding"]

        result = index.query(
            vector=embed,
            top_k=payload.top_k,
            include_metadata=True,
            filter=payload.filters or {}
        )

        return JSONResponse(content={"results": result["matches"]})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
