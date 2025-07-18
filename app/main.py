# main.py
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import logging, uvicorn
from app.kg_builder import KnowledgeGraphBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Langtron Medical KG API",
    description="Transform any clinical document into a Neo4j knowledge graph.",
    version="1.1.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

builder = KnowledgeGraphBuilder()

@app.post("/upload")
async def upload(file: UploadFile):
    tmp_path = Path(f"temp_{file.filename}")
    try:
        tmp_path.write_bytes(await file.read())
        doc_id = builder.process_document(tmp_path)
        return {"doc_id": doc_id}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)

@app.get("/entities/{doc_id}")
def entities(doc_id: str):
    return {"entities": builder.get_entities(doc_id)}

@app.get("/relations/{doc_id}")
def relations(doc_id: str):
    return {"relations": builder.get_relations(doc_id)}

@app.get("/graph/{doc_id}")
def graph(doc_id: str):
    return {"graph": builder.get_graph(doc_id)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)