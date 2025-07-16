from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import openai
import os
from pinecone import Pinecone  # ✅ new SDK import

app = FastAPI()

# ✅ Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://vercel.com"],  # ✅ Your actual frontend domain
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],          # ✅ Commonly used methods for APIs
    allow_headers=["Content-Type", "Authorization"],   # ✅ Typical headers needed for POST & auth
)


# ✅ Load API keys from environment
openai.api_key = os.environ["OPENAI_API_KEY"]
pinecone_api_key = os.environ["PINECONE_API_KEY"]
pinecone_index_name = os.environ["PINECONE_INDEX_NAME"]

# ✅ Initialize Pinecone client and index
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# ✅ Define input model
class QueryRequest(BaseModel):
    query: str
    filters: Optional[Dict] = None
    top_k: Optional[int] = 5

# ✅ MCP search endpoint
@app.post("/mcp")
async def mcp_search(payload: QueryRequest, request: Request):
    try:
        # Generate OpenAI embedding
        embedding_response = openai.Embedding.create(
            input=payload.query,
            model="text-embedding-3-small"
        )
        embed = embedding_response["data"][0]["embedding"]

        # Query Pinecone
        result = index.query(
            vector=embed,
            top_k=payload.top_k,
            include_metadata=True,
            filter=payload.filters or {}
        )

        return JSONResponse(content={"results": result["matches"]})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
