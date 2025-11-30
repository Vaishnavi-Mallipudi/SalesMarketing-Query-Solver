
import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
import pandas as pd
from datetime import datetime
import csv

st.set_page_config(page_title="Sales & Marketing Support Agent", layout="wide")
st.title("Sales & Marketing Support Agent — Demo")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("Set your OPENAI_API_KEY in Streamlit secrets or environment variables.")
    st.stop()

# initialize LLM and embeddings
llm = ChatOpenAI(temperature=0.1, openai_api_key=OPENAI_API_KEY)
embedder = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# connect to local Chroma (ensure ingest.py was run first to populate ./chroma_db)
persist_dir = "./chroma_db"
try:
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedder)
    retriever = vectordb.as_retriever(search_kwargs={"k":4})
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever, return_source_documents=True)
except Exception as e:
    st.error(f"Could not connect to Chroma vector DB. Run ingest.py first. Error: {e}")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar: Lead capture & social generator
st.sidebar.header("Lead Capture")
name = st.sidebar.text_input("Name")
email = st.sidebar.text_input("Email")
interest = st.sidebar.text_input("Interest / Product")
budget = st.sidebar.text_input("Budget (optional)")
if st.sidebar.button("Save lead"):
    leads_file = "leads.csv"
    score = 0
    try:
        b = float(budget)
        if b > 100: score += 30
    except:
        pass
    if interest:
        if any(k in interest.lower() for k in ["buy","purchase","order","need","urgent"]):
            score += 30
    if email:
        score += 10
    # interactions heuristic
    score += min(len(st.session_state.history)*10, 30)
    status = "cold"
    if score >= 70: status = "hot"
    elif score >= 40: status = "warm"

    row = [datetime.utcnow().isoformat(), name, email, interest, budget, score, status]
    header = ["timestamp","name","email","interest","budget","lead_score","status"]
    write_header = not os.path.exists(leads_file)
    with open(leads_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)
    st.sidebar.success(f"Saved lead (score={score}, status={status})")

st.sidebar.header("Social Post Generator")
sp_product = st.sidebar.text_input("Product for caption")
sp_audience = st.sidebar.text_input("Target audience (e.g., photographers)")
if st.sidebar.button("Generate captions"):
    if not sp_product:
        st.sidebar.error("Enter a product name")
    else:
        prompt = f"""Create 3 short social media captions for the product: {sp_product}.
        Audience: {sp_audience or 'general'}. Keep each caption to 1-2 short sentences and include a call-to-action."""
        resp = llm.generate([{"role":"user","content":prompt}])
        text = resp.generations[0][0].text if hasattr(resp, 'generations') else str(resp)
        st.sidebar.write(text)

# Main: Chat
st.subheader("Support chat / product Q&A")
query = st.text_input("Ask a product or FAQ question (try: 'What is the return policy?')", key="input")
if st.button("Send"):
    if not query.strip():
        st.warning("Type a question first.")
    else:
        result = qa_chain({"question": query, "chat_history": st.session_state.history})
        answer = result.get("answer","(no answer)")
        st.markdown("**Agent:** " + answer)
        st.write("Sources:")
        for d in result.get("source_documents", []):
            st.write("-", d.metadata.get("source","unknown"))
        # append to history
        st.session_state.history.append((query, answer))
