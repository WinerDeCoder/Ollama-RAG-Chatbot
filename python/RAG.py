from langchain_community.document_loaders import WebBaseLoader
from langchain_community.retrievers import TavilySearchAPIRetriever
import asyncio
from bs4 import BeautifulSoup
from ollama import AsyncClient
from ollama import chat
from ollama import Client
import re
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

def clean_text(text):
    """Remove all characters except letters (including Vietnamese), numbers, and spaces."""
    return re.sub(r'[^a-zA-Z0-9\s\u00C0-\u1EF9]', '', text)

web_retriever = TavilySearchAPIRetriever(k=2)

client_main = Client(
  host='http://localhost:11434',
  headers={'x-some-header': 'some-value'}
)

prompt_filter = """You are a good post reader and filtering and summary. 
Given content of a website - that supposed to be relevent to the query - however, this content anything on the website, not only the main content. 
Therefore, you need to filter out only relevant information, main content of the website to the query and summary them

The web content: {context}
        
The query: {question}

Attention: 
- Do not answer the query
- The content should only contain knowledge, avoid owner information, phone number, email,....
- The website is usually Vietnamses, so if yes, answer by Vietnamese
- Remember to summary the content and return this summary
"""


prompt_answer = """You are an AI assistant that will help answer the question of user based on given information
The given information related to the user's question: {context}

Attention:
- Answer friendly and correctly
- Always use the given information but if they are not good, add more information with your own knowledge
- Just answer long enough and take main, core information, should not surpass 500 words
- Always answer in Vietnamese
"""
        


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


def main():
    question = input("Enter your question: ")
    
    results = web_retriever.invoke(question)
    
    # Run the async process
    synthetic_information = asyncio.run(synthetic_doc(results, prompt_filter, question))
    
    message = prompt_answer.format(context = synthetic_information)
    

    stream = client_main.chat(
        model='gemma3:12b',
        messages=[{'role': 'system', 'content': message}, 
                {'role': 'user'  , 'content': question}]
    )

    print(stream["message"]["content"])
    
if __name__ == "__main__":
    while True:
        main()
