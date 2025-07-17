from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import os
from openai import OpenAI
from pinecone import Pinecone

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")

print("🔍 Pinecone API Key:", "SET" if pinecone_api_key else "NOT SET")
print("🔍 Pinecone Index Name:", pinecone_index_name)

if not pinecone_api_key or not pinecone_index_name:
    raise RuntimeError("Missing Pinecone API key or index name in environment variables.")

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(name=pinecone_index_name)
print("🔍 Index object:", index)

class QueryRequest(BaseModel):
    query: str
    filters: Optional[Dict] = None
    top_k: Optional[int] = 5

@app.post("/mcp")
async def mcp_search(payload: QueryRequest, request: Request):
    try:
        print("📩 Incoming query:", payload.query)
        print("📂 Filters:", payload.filters)
        print("🔝 Top K:", payload.top_k)

        response = client.embeddings.create(
            input=payload.query,
            model="text-embedding-3-small"
        )
        embed = response.data[0].embedding
        print("🧠 Embedding generated:", embed[:5], "...")

        if index is None:
            raise HTTPException(status_code=500, detail="Pinecone index is not initialized")

        result = index.query(
            vector=embed,
            top_k=payload.top_k,
            include_values=True,
            filter=payload.filters or {}
        )

        # print("✅ Query result:", result)

        matches = result.matches or []
        return JSONResponse(content={
            "results": [
                {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                }
                for match in matches
            ]
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
