__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_ollama import OllamaEmbeddings
import chromadb
from uuid import uuid4
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import torch

import logging
logging.getLogger().setLevel(logging.ERROR)

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    'jinaai/jina-reranker-v2-base-multilingual',
    torch_dtype=torch.float16,
    trust_remote_code=True,
    use_flash_attn=False
)

model.to("cuda:0")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_db_summary = chroma_client.get_or_create_collection(name="summary_bge", metadata={"hnsw:space": "cosine"})
chroma_db_full_document = chroma_client.get_or_create_collection(name="full_document_bge", metadata={"hnsw:space": "cosine"})

# Initialize Ollama Embeddings (using a model like "mistral" or "llama3")
embedding_model = OllamaEmbeddings(model="bge-m3:latest")  # Change model as needed

# Initialize FastAPI
app = FastAPI()

# Pydantic model for input
class QueryRequest(BaseModel):
    question: str
    num_query: int
    top_n: int
    
# Pydantic model for input
class Response(BaseModel):
    documents: str
    match_summary: List[str]

@app.post("/api/chat")
async def chat(request: QueryRequest):
    
    try:
        question = request.question
        top_n    = request.top_n
        num_query = request.num_query
        
        print(question)

        # Retrieve relevant documents
        query_embedding = embedding_model.embed_documents([question])
        
        chroma_db_summary.get()

        matched_summary = chroma_db_summary.query(
            query_embeddings=query_embedding,
            n_results=num_query
        )
        
        matched_full_document = chroma_db_full_document.query(
            query_embeddings=query_embedding,
            n_results=num_query
        )
        
        summary_ids, summary_documents, summary_distances = matched_summary['ids'], matched_summary['documents'], matched_summary['distances']
        
        full_documenty_ids, full_document_documents, full_document_distances = matched_full_document['ids'], matched_full_document['documents'], matched_full_document['distances']
        
        #Take missture from summary and full doc
        id_list = []
        distances_list = []
        
        for summary_id, full_documenty_id, summary_distance, full_document_distance in zip(summary_ids[0], full_documenty_ids[0], summary_distances[0], full_document_distances[0]):
            if len(id_list) < num_query  :
                if summary_id == full_documenty_id and summary_id not in id_list:
                    id_list.append(summary_id)
                    distances_list.append(summary_distance)
                else:
                    if summary_id not in id_list:
                        id_list.append(summary_id)
                        distances_list.append(summary_distance)
                    
                    if full_documenty_id not in id_list and len(id_list) < num_query:
                        id_list.append(full_documenty_id)
                        distances_list.append(full_document_distance)
            else:
                break
        
        data = []
        sentence_pairs = []

        # Step 1: Fetch content and prepare sentence pairs
        for id, distance in zip(id_list, distances_list):
            full_content = chroma_db_full_document.get(ids=id)
            doc_text = full_content["documents"][0]

            sentence_pairs.append([question, doc_text])
            data.append({
                "full_content": doc_text,
                "distance": distance
            })

        # Step 2: Compute rerank scores
        scores = model.compute_score(sentence_pairs)

        # Step 3: Combine everything and sort by score
        scored_data = [
            {
                "full_content": item["full_content"],
                "distance": item["distance"],
                "score": float(score)
            }
            for item, score in zip(data, scores)
        ]

        scored_data.sort(key=lambda x: x["score"], reverse=True)

        # Step 4: Keep only top N
        top_data = scored_data[:top_n]
                  
        return JSONResponse(content={"status": "success", "data": top_data}, status_code=200)
    
    except Exception as e:
        
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


class Data_Item(BaseModel):
    summary: str
    full_content: str
    

@app.post("/api/add-data")
async def chat(request: Data_Item):
    
    try:

        summary = request.summary

        full_content = request.full_content
        uuid = str(uuid4())

        
        embedding_vectors = embedding_model.embed_documents([summary, full_content])

        chroma_db_summary.add(
            documents=[summary],
            embeddings=[embedding_vectors[0]],
            ids= [uuid]
        )

        chroma_db_full_document.add(
            documents=[full_content],
            embeddings=[embedding_vectors[-1]],
            ids= [uuid]
        )

        
        return JSONResponse(content={"status": "success", "uui": uuid}, status_code=200)
    
    except Exception as e:
        
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
    
    
#EndPoint to retrieve all data
@app.get("/api/get-all-data")
def get_all_data():
    try:
        # Get all data from ChromaDB
        data = chroma_db_summary.get()

        # Ensure we have the correct fields
        documents = data.get("documents", [])
        uuids     = data.get("ids", [])
        
        full_contents = []
        for uuid in uuids:  
            retrieves = chroma_db_full_document.get(ids = [uuid])
            full_contents.append(retrieves['documents'][0])
        
        # Pair documents with their corresponding UUIDs
        paired_data = [
            {"uuid": uuid, "document": doc, "full_content": full_content}
            for uuid, doc, full_content  in zip(uuids, documents, full_contents)
        ]

        return JSONResponse(content={"status": "success", "total_item": len(full_contents), "data": paired_data}, status_code=200)
    
    except Exception as e:
        
        return JSONResponse(content={"status": "failed", "message": str(e)}, status_code=500)
    
    
# Request model for updating documents
class UpdateRequest(BaseModel):
    uuid: str
    summary: str
    full_content: str

@app.put("/api/update-data")
def update_data(request: UpdateRequest):
    try:
        
        question_id = chroma_db_summary.get(ids = [request.uuid])
        
        if len(question_id['ids']) ==0:
            return JSONResponse(content={"status": "Invalid", "message": "UUID not found", "uuid": request.uuid}, status_code=404)
        
        
        embedding_vectors = embedding_model.embed_documents([request.summary, request.full_content])
        
        # Perform update in ChromaDB
        chroma_db_summary.update(
            ids=[request.uuid],
            embeddings=[embedding_vectors[0]],
            documents=[request.summary]
        )
        
        # Perform update in ChromaDB
        chroma_db_full_document.update(
            ids=[request.uuid],
            embeddings=[embedding_vectors[-1]],
            documents=[request.full_content]
        )

        return JSONResponse(content={"status": "success", "message": "Data updated successfully", "uuid": request.uuid}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"status": "failed", "message": str(e), "uuid": request.uuid}, status_code=500)
    
    
#Endpoint to delete ids
# Define request body model
class DeleteIDsRequest(BaseModel):
    uuid: str # List of IDs to delete

@app.delete("/api/delete-item")
def delete_by_ids(request: DeleteIDsRequest):
    try:
        
        question_id = chroma_db_summary.get(ids = [request.uuid])
        
        if len(question_id['ids']) ==0:
            return JSONResponse(content={"status": "Invalid", "message": "UUID not found", "uuid": request.uuid}, status_code=404)
        

        # Delete the requested IDs from ChromaDB
        chroma_db_summary.delete(ids=[request.uuid])
        
        chroma_db_full_document.delete(ids=[request.uuid])

        return JSONResponse(
            content={"status": "success", "message": "Deleted successfully", "uuid": request.uuid},
            status_code=200
        )

    except Exception as e:
        print(f"Error deleting IDs: {e}")
        return JSONResponse(
            content={"status": "failed", "message": str(e)},
            status_code=500
        )
