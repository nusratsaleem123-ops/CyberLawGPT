import os
import requests
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Configuration & Constants
PDF_URL = "https://drive.google.com/uc?export=download&id=1UG3QhN70CfcKCh7kJPbZHFpmbwvvNP0X"
PDF_PATH = "PECA_Act_2016.pdf"

st.set_page_config(
    page_title="CyberlawsGPT - Pakistani Cyber Law Assistant",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ CyberlawsGPT")
st.markdown("Ask questions about the **Prevention of Electronic Crimes Act (PECA), 2016** of Pakistan.")

# Sidebar Controls
st.sidebar.header("⚙️ Configuration & Model Settings")

# Grok API Key
grok_api_key = st.sidebar.text_input(
    "xAI / Grok API Key",
    type="password",
    value=os.getenv("XAI_API_KEY", ""),
    help="Enter your xAI Grok API key (starts with xai-)."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Customize Responses")

# Technical / Legal Depth Option
tech_level = st.sidebar.select_slider(
    "Technical / Legal Level",
    options=["Layman / Simple", "General Public", "Legal Student", "Lawyer / Expert"],
    value="General Public",
    help="Controls the technical complexity of the language."
)

# Response Length Option
response_size = st.sidebar.select_slider(
    "Response Size",
    options=["Brief (1-2 Paras)", "Standard Summary", "Detailed Breakdown", "Exhaustive Analysis"],
    value="Standard Summary",
    help="Controls how concise or detailed the output is."
)

# Persona / Tone Option
persona = st.sidebar.selectbox(
    "Perspective / Role",
    options=[
        "Neutral Legal Assistant",
        "Defense Attorney Viewpoint",
        "Public Prosecutor Viewpoint",
        "Compliance Officer / Enterprise"
    ],
    index=0,
    help="Frame the answer from a specific legal angle."
)

# Creativity Option
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.1,
    help="Lower values produce more direct, predictable answers."
)

# Retrieval Chunk Count
top_k = st.sidebar.slider("Retrieved Chunks (k)", min_value=2, max_value=10, value=4)

# Download PDF function
def download_pdf(url, dest_path):
    if not os.path.exists(dest_path):
        with st.spinner("Downloading PECA Act 2016 PDF..."):
            res = requests.get(url, allow_redirects=True)
            if res.status_code == 200:
                with open(dest_path, "wb") as f:
                    f.write(res.content)
                st.success("PDF downloaded successfully!")
            else:
                st.error("Failed to download PDF document from Google Drive.")

# Load Embeddings & Build Vector Index
@st.cache_resource(show_spinner="Indexing PECA Act text into FAISS vector store...")
def init_vector_store():
    download_pdf(PDF_URL, PDF_PATH)
    
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(splits, embeddings)
    return vector_store

# Initialize App Vector Store
try:
    vector_store = init_vector_store()
    st.sidebar.success("✅ FAISS Vector Index Loaded")
except Exception as e:
    st.error(f"Failed to initialize vector database: {e}")
    st.stop()

# System Prompt Template
system_prompt_str = f"""You are CyberlawsGPT, an expert AI assistant specializing in Pakistani Cyber Law, specifically the Prevention of Electronic Crimes Act (PECA), 2016.

Adhere strictly to the following parameters set by the user:
- Target Technical Level: {tech_level}
- Desired Response Size: {response_size}
- Perspective / Role: {persona}

Answer questions based strictly on the provided context extracted from PECA 2016. Always cite relevant Sections, Chapters, or Penalties where applicable. If the context does not contain enough information to answer, state clearly what the law provides or note that it is absent from the text.

Context:
{{context}}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_str),
    ("human", "{input}")
])

# Interface Controls
st.subheader("💬 Ask CyberlawsGPT")

preset_query = st.selectbox(
    "Quick Sample Questions:",
    [
        "Custom Question...",
        "What is the penalty for unauthorized access to an information system or data?",
        "How does PECA 2016 address cyberstalking and cyberbullying?",
        "What constitutes cyber terrorism under Section 10?",
        "What provisions are made for the protection of children online?",
        "What are the powers of the designated investigation agency?"
    ]
)

user_query = st.text_input("Type your question here:", value="" if preset_query == "Custom Question..." else preset_query)

if st.button("Submit Query", type="primary"):
    if not grok_api_key:
        st.warning("⚠️ Please provide your xAI / Grok API key in the sidebar.")
    elif not user_query.strip():
        st.warning("⚠️ Please enter a question to ask.")
    else:
        try:
            with st.spinner("Processing document references via Grok API..."):
                # Connect Grok model via xAI ChatOpenAI endpoint
                llm = ChatOpenAI(
                    model="grok-beta",
                    api_key=grok_api_key,
                    base_url="https://api.x.ai/v1",
                    temperature=temperature
                )
                
                retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
                combine_docs_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
                
                response = rag_chain.invoke({"input": user_query})
                
                st.markdown("### 📜 Answer")
                st.write(response["answer"])
                
                with st.expander("🔍 View Referenced Context Chunks"):
                    for idx, doc in enumerate(response["context"], 1):
                        page = doc.metadata.get("page", "Unknown")
                        st.markdown(f"**Source Chunk {idx} (PDF Page {page}):**")
                        st.caption(doc.page_content)
                        st.divider()
        except Exception as err:
            st.error(f"Error querying Grok API: {err}")

st.markdown("---")
st.caption("CyberlawsGPT is an automated information system for PECA 2016 and does not constitute official legal advice.")
