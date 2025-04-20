from fastapi import FastAPI, HTTPException, Query
from typing import Optional
from pydantic import BaseModel, Field
from typing import Literal
import requests
import os
import re
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

# Load environment variables
load_dotenv()
JINA_AI_API_KEY = os.getenv("JINA_AI_API_KEY")

JINA_API_URL = "https://s.jina.ai/"
HEADERS = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {JINA_AI_API_KEY}',
    'X-No-Cache': 'true',
    "X-Retain-Images": "none",
    "X-With-Links-Summary": "true",
    "X-Return-Format": "text"
}

model = CrossEncoder(
    "jinaai/jina-reranker-v2-base-multilingual",
    automodel_args={"torch_dtype": "auto"},
    trust_remote_code=True,
)

model.to("cuda:0") # or "cpu" if no GPU is available

def rerank_content(query, content):
    url = 'https://api.jina.ai/v1/rerank'

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer jina_0164761bfb794356bf1658bc77fbe2515XdbSnoabCF_uFdxVM0WxA8UlEOQ'
    }

    data = {
        "model": "jina-reranker-v2-base-multilingual",
        "query": "Làm sao để phòng trừ ruồi đục lá ?",
        "top_n": 3,
        "documents": content.split("\n\n")
    }

    response = requests.post(url, headers=headers, json=data)

    max_index = max([index['index'] for index in response.json()['results']])

    index_list = ["" for _ in range(max_index+1)]

    for item in response.json()['results']:
        index = item['index']
        if index !=0 and index != max_index:
            index_list[index-1] = content.split("\n\n")[index-1]
            index_list[index+1] = content.split("\n\n")[index+1]
        index_list[index] = item['document']['text'] + "\n\n"


    merged_text = "".join(index_list)
        
    return merged_text


app = FastAPI()

# Pydantic model for input
class QueryRequest(BaseModel):
    query: str
    num_retrieve: int

@app.post("/search")
def search(query: QueryRequest):
    params = {
        "q": query.query,
        "num": query.num_retrieve
    }

    response = requests.get(JINA_API_URL, headers=HEADERS, params=params)
    
    print(response.text)
    
    full_response = response.json()

    data_list = []

    for item in full_response['data']:
        data_list.append({
            "title": item.get("title", ""),
            "url": item.get('url', ""),
            "description": item.get("description", ""),
            "text": rerank_content(item.get("text", "")),
        })

    return data_list