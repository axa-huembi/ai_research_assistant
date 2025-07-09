import os
import requests
import faiss


from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

from bs4 import BeautifulSoup

os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"


def gather_information(topic):
 url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
 response = requests.get(url)
 soup = BeautifulSoup(response.content, 'html.parser')
 paragraphs = soup.find_all('p')
 text = ' '.join([p.get_text() for p in paragraphs])
 return text
def truncate_text(text, max_tokens=2000):
 """ Truncate text to approximately max_tokens. """
 words = text.split()
 return ' '.join(words[:max_tokens])
def analyze_information(info):
 llm = OpenAI(temperature=0.7, max_tokens=500) # Limit the response tokens
 truncated_info = truncate_text(info, max_tokens=1500) # Further reduce input tokens
 prompt = f"Summarize key points from this text:\n\n{truncated_info}\n\nKey points:"
 response = llm(prompt)
 return response
def generate_summary(analysis):
 llm = OpenAI(temperature=0.7)
 prompt = f"""
 Based on the following analysis, generate a concise summary:
 {analysis}

 Concise summary:
 """
 summary = llm(prompt)
 return summary
def create_knowledge_base(text):
 text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
 texts = text_splitter.split_text(text)
 embeddings = OpenAIEmbeddings()
 knowledge_base = FAISS.from_texts(texts, embeddings)
 return knowledge_base
def query_knowledge_base(query, kb):
 docs = kb.similarity_search(query, k=1)
 return docs[0].page_content
# Example usage
if __name__ == "__main__":
 topic = "Artificial Intelligence"
 info = gather_information(topic)
 analysis = analyze_information(info)
 summary = generate_summary(analysis)
 print(f"Summary: {summary}")

 kb = create_knowledge_base(info)
 query = "What are the main applications of AI?"
 result = query_knowledge_base(query, kb)
 print(f"Query result: {result}")