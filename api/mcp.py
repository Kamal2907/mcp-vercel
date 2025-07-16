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

# ✅ Initialize OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ✅ Initialize Pinecone client
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")

print("🔍 Pinecone API Key:", "SET" if pinecone_api_key else "NOT SET")
print("🔍 Pinecone Index Name:", pinecone_index_name)

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)
print("🔍 Index object:", index)

# ✅ Request body model
class QueryRequest(BaseModel):
    query: str
    filters: Optional[Dict] = None
    top_k: Optional[int] = 5

# ✅ POST endpoint for /mcp
@app.post("/mcp")
async def mcp_search(payload: QueryRequest, request: Request):
    try:
        print("📩 Incoming query:", payload.query)
        print("📂 Filters:", payload.filters)
        print("🔝 Top K:", payload.top_k)

        # ✅ Generate embeddings using new SDK
        response = client.embeddings.create(
            input=payload.query,
            model="text-embedding-3-small"
        )
        embed = response.data[0].embedding
        print("🧠 Embedding generated:", embed[:5], "...")  # Print first 5 values for brevity

        # ✅ Check if index is valid
        if index is None:
            print("❌ Index is None!")
            raise HTTPException(status_code=500, detail="Pinecone index is not initialized")

        # ✅ Query Pinecone
        result = index.query(
            vector=embed,
            top_k=payload.top_k,
            include_metadata=True,
            filter=payload.filters or {}
        )

        print("✅ Query result:", result)

        return JSONResponse(content={
            "results": [match.dict() for match in result.matches]
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
