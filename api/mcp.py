from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict
import openai
import pinecone
import os

app = FastAPI()

# ENV Vars
openai.api_key = os.environ["OPENAI_API_KEY"]
pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"])
index = pinecone.Index(os.environ["PINECONE_INDEX_NAME"])

class QueryRequest(BaseModel):
    query: str
    filters: Optional[Dict] = None
    top_k: Optional[int] = 5

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

