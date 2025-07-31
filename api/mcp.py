from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import os
import httpx
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

# GA4 Measurement Protocol setup
GA4_MEASUREMENT_ID = os.environ.get("GA4_MEASUREMENT_ID")
GA4_API_SECRET = os.environ.get("GA4_API_SECRET")
GA4_ENDPOINT = "https://www.google-analytics.com/mp/collect"

async def log_to_google_analytics(query: str, ip: str = "0.0.0.0"):
    if not GA4_MEASUREMENT_ID or not GA4_API_SECRET:
        print("❌ GA4 credentials not set")
        return

    payload = {
        "client_id": "api_user",
        "events": [
            {
                "name": "mcp_api_request",
                "params": {
                    "query": query,
                    "ip_override": ip,
                    "source": "FastAPI"
                }
            }
        ]
    }

    params = {
        "measurement_id": GA4_MEASUREMENT_ID,
        "api_secret": GA4_API_SECRET
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(GA4_ENDPOINT, params=params, json=payload)
            print(f"📊 GA4 logged: {response.status_code}")
    except Exception as e:
        print("❌ Error sending to GA4:", e)

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

        # Log to GA4
        await log_to_google_analytics(query=payload.query, ip=request.client.host)

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
            include_metadata=True,
            filter=payload.filters or {}
        )

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
