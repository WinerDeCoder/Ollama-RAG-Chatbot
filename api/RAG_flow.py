from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
from ollama import AsyncClient, Client
import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Initialize FastAPI
app = FastAPI()

# Initialize clients
web_retriever = TavilySearchAPIRetriever(k=2)
client_main = Client(host='http://localhost:11434')

# Define prompts
prompt_filter = """You are a good post reader and filtering and summary. 
Given content of a website - that supposed to be relevant to the query - however, this content contains anything on the website, not only the main content. 
Therefore, you need to filter out only relevant information, main content of the website to the query and summarize them.

The web content: {context}
The query: {question}

Attention: 
- Do not answer the query
- The content should only contain knowledge, avoid owner information, phone number, email,....
- The website is usually Vietnamese, so if yes, answer in Vietnamese
- Remember to summarize the content and return this summary.
"""

prompt_answer = """You are an AI assistant that will help answer the question of the user based on given information.

The given information related to the user's question: {context}

Attention:
- Answer friendly and correctly.
- Always use the given information but if they are not good, add more information with your own knowledge.
- Just answer long enough and take the main, core information, should not surpass 500 words.
- Always answer in Vietnamese.
"""

# Pydantic model for input
class QueryRequest(BaseModel):
    question: str

async def process_document(document, client, prompt, question):
    """Load a webpage, clean its content, and generate synthetic information using an AI model."""
    source = document.metadata["source"]
    print(f"Processing: {source}")

    # Load webpage
    loader = WebBaseLoader(web_paths=(source,))
    blog_docs = loader.load()

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(blog_docs[0].page_content, "html.parser")
    cleaned_text = re.sub(r'\n+', '\n', str(soup))

    # Format prompt
    message = prompt.format(context=cleaned_text, question=question)

    # Async call to Ollama
    response = await client.chat(
        model='gemma3:4b',
        messages=[{'role': 'system', 'content': message}]
    )

    return response["message"]["content"]

async def synthetic_doc(results, prompt, question):
    client = AsyncClient()
    
    # Run all tasks asynchronously
    tasks = [process_document(doc, client, prompt, question) for doc in results]
    responses = await asyncio.gather(*tasks)

    # Combine all responses
    synthetic_information = "\n\n".join(responses)
    return synthetic_information

@app.post("/chat")
async def chat(request: QueryRequest):
    question = request.question

    # Retrieve relevant documents
    results = web_retriever.invoke(question)
    
    # Generate synthetic information
    synthetic_information = await synthetic_doc(results, prompt_filter, question)
    
    message = prompt_answer.format(context=synthetic_information)

    # Get response from Ollama
    stream = client_main.chat(
        model='gemma3:12b',
        messages=[{'role': 'system', 'content': message}, {'role': 'user', 'content': question}]
    )

    return {"response": stream["message"]["content"]}

# Run with: `uvicorn rag_api:app --host 0.0.0.0 --port 8000`
