from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict
import openai
import os
from pinecone import Pinecone  # ✅ new SDK import

app = FastAPI()

# Load ENV variables
openai.api_key = os.environ["OPENAI_API_KEY"]

# ✅ NEW Pinecone client instance (DO NOT use pinecone.init)
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# ✅ Get the existing index
index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

class QueryRequest(BaseModel):
    query: str
    filters: Optional[Dict] = None
    top_k: Optional[int] = 5

# ✅ Route MUST match your vercel.json ("/" if src is "/mcp")
@app.post("/")
async def mcp_search(payload: QueryRequest, request: Request):
    try:
        # Generate embedding from OpenAI
        embed = openai.Embedding.create(
            input=payload.query,
            model="text-embedding-3-small"
        )["data"][0]["embedding"]

        # Query Pinecone index
        result = index.query(
            vector=embed,
            top_k=payload.top_k,
            include_metadata=True,
            filter=payload.filters or {}
        )

        return JSONResponse(content={"results": result["matches"]})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
