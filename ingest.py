
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import CSVLoader
from langchain.schema import Document
import pandas as pd

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Set OPENAI_API_KEY environment variable before running.")
    raise SystemExit(1)

emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Load product CSV and FAQ CSV and create simple documents
products = pd.read_csv("sample_products.csv")
faqs = pd.read_csv("sample_faqs.csv")

docs = []
for _, r in products.iterrows():
    content = f"Title: {r['title']}\nDescription: {r['description']}\nCategory: {r['category']}\nPrice: {r['price']}\nTags: {r['tags']}"
    docs.append({"page_content": content, "metadata": {"source": f"product:{r['id']}"}})

for _, r in faqs.iterrows():
    content = f"Q: {r['question']}\nA: {r['answer']}"
    docs.append({"page_content": content, "metadata": {"source": f"faq:{r['id']}"}})

persist_dir = "./chroma_db"
vectordb = Chroma.from_documents(docs, embedding=emb, persist_directory=persist_dir)
vectordb.persist()
print(f"Persisted {len(docs)} documents to Chroma at {persist_dir}")
