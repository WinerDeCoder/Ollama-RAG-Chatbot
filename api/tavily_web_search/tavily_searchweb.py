from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from tavily import TavilyClient
from transformers import AutoModelForSequenceClassification
import torch
from multiprocessing import Pool
from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv(".env")

model = AutoModelForSequenceClassification.from_pretrained(
    "jinaai/jina-reranker-v2-base-multilingual",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    use_flash_attn=False
)

model.to("cuda:0")

# API key mặc định, bạn có thể thay bằng env variable nếu cần
TAVILY_API_KEY =  os.getenv("TAVILY_API_KEY")

# Khởi tạo Tavily Client
client = TavilyClient(api_key=TAVILY_API_KEY)


def process_document(args):
    query, content, top_n = args
    if content is None:
        item_split = ["No content"]
    else:
        item_split = [doc for doc in content.split("\n\n") if len(doc) > 100]
    
    if not item_split:
        return ""
    
    if top_n ==0:
        return "".join(item_split)
    
    # Construct sentence pairs
    sentence_pairs = [[query, doc] for doc in item_split]
    
    # Compute scores
    scores = model.compute_score(sentence_pairs)
    
    # Combine documents with scores and original indices
    scored_docs = list(zip(item_split, scores, range(len(item_split))))
    
    # Sort by score descending
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    try:
        # Get top N results
        top_results = scored_docs[:top_n]
        
        if max([len(text) for text in top_results]) < 400:
            top_results = scored_docs[:top_n+int(top_n/2)]
    except:
        top_results = scored_docs
    
    # Sort top results by original index to maintain order
    top_results.sort(key=lambda x: x[2])
    
    # Merge top documents
    merged_text = "".join(doc for doc, _, _ in top_results)
    
    return merged_text

def rerank_content(query: str, content: list, top_n: int = 8):
    with Pool() as pool:
        args = [(query, doc, top_n) for doc in content]
        list_result = pool.map(process_document, args)
    return list_result



# Yêu cầu đầu vào
class TavilySearchRequest(BaseModel):
    query: str = Field(..., description="Câu hỏi hoặc từ khoá cần tìm kiếm")
    search_type: Literal["regular", "context", "qa"] = Field(
        "regular", description="Loại tìm kiếm: regular, context, hoặc qa"
    )
    search_depth: Literal["basic", "advanced"] = Field(
        "basic", description="Mức độ tìm kiếm (basic hoặc advanced)"
    )
    include_answer: bool = Field(
        True, description="Có trả về câu trả lời AI không?"
    )
    max_results: int = Field(
        3, ge=1, le=10, description="Số kết quả tối đa (1–10)"
    )
    rerank_top_n: int = Field(
        4, ge=0, le=15, description="Lấy top n rerank tối đa (1–15), if 0 mean not rerank, take full"
    )

app = FastAPI()

@app.post("/search")
async def tavily_search(request: TavilySearchRequest):
    try:
        if request.search_type == "context":
            result = client.get_search_context(query=request.query)
            return {
                "type": "context",
                "query": request.query,
                "result": result,
            }

        elif request.search_type == "qa":
            result = client.qna_search(query=request.query)
            return {
                "type": "qa",
                "query": request.query,
                "answer": result,
            }

        else:  # regular
            result = client.search(
                query=request.query,
                max_results= request.max_results,
                include_raw_content=True
            )
            #print(result['results'])
            
            # print(request.max_results)
            
            #print(len(result['results']))

            # Tùy chỉnh format kết quả
            formatted_results = {
                "type": "regular",
                "query": request.query,
                "data": [],
            }
            
            combine_raw_content = [item["raw_content"] for item in result['results'][:request.max_results]]
            
            
            #print(len(combine_raw_content))
            
            combine_rerank_content = rerank_content(request.query, combine_raw_content, top_n = request.rerank_top_n)

            for item, rerank_doc in zip(result.get("results", [])[:request.max_results], combine_rerank_content):
                formatted_results["data"].append({
                    "title": item.get("title", "No title"),
                    "url": item.get("url", "No URL"),
                    #"score": item.get("score", 0),
                    "content": item.get("content", ""),
                    "raw_content": rerank_doc,
                })

            return formatted_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi truy vấn Tavily: {str(e)}")
