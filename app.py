import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Zyro Dynamics HR Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Zyro Dynamics HR Chatbot")
st.markdown("Ask questions about company HR policies.")

# -----------------------------
# LOAD LLM
# -----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    groq_api_key=GROQ_API_KEY
)

# -----------------------------
# BUILD RAG
# -----------------------------
@st.cache_resource
def load_rag():

    loader = PyPDFDirectoryLoader("./hr_docs")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    return retriever

retriever = load_rag()

# -----------------------------
# RAG PROMPT
# -----------------------------
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an HR assistant for Zyro Dynamics.

Answer only using the provided context.

Context:
{context}

Question:
{question}

If the answer is not available in the context,
reply:

"I could not find that information in the HR policies."

Answer:
""")

# -----------------------------
# GUARDRAILS
# -----------------------------
OOS_PROMPT = ChatPromptTemplate.from_template("""
Determine whether the question is related to
Zyro Dynamics HR policies, employee handbook,
leave policy, benefits, travel, performance review,
work from home, onboarding, compensation, security
or other company documents.

Question:
{question}

Answer ONLY YES or NO.
""")

REFUSAL_MESSAGE = """
I can only answer questions related to Zyro Dynamics HR policies and company documents.
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question):

    docs = retriever.invoke(question)

    context = format_docs(docs)

    chain = (
        RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain.invoke({
        "context": context,
        "question": question
    })

def ask_bot(question):

    guardrail_chain = (
        OOS_PROMPT
        | llm
        | StrOutputParser()
    )

    decision = guardrail_chain.invoke({
        "question": question
    }).strip().upper()

    if "NO" in decision:
        return REFUSAL_MESSAGE

    return rag_chain(question)

# -----------------------------
# CHAT UI
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask your HR question")

if question:

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_bot(question)

        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
