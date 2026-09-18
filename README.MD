# ⚖️ CyberlawsGPT

CyberlawsGPT is a Retrieval-Augmented Generation (RAG) web interface built with **Streamlit**, **FAISS**, **LangChain**, and **xAI's Grok API** to answer queries regarding Pakistan's **Prevention of Electronic Crimes Act (PECA), 2016**.

---

## ⚡ Key Capabilities

- **Automatic PDF Ingestion:** On app startup, the official PECA 2016 Act PDF is fetched directly from Google Drive, split into semantic text chunks, and stored in a local FAISS vector index.
- **Grok LLM Integration:** Uses the xAI `grok-beta` API model for natural language generation.
- **Customizable UI Controls:**
  - **Technical Level:** Adjust language between simple explanations and formal legal citations.
  - **Response Size:** Toggle output length from brief paragraphs to exhaustive analyses.
  - **Perspective Tuning:** View answers from a Neutral, Defense, Prosecution, or Corporate Compliance angle.
  - **Temperature Adjustment:** Fine-tune response determinism.

---

## 🛠️ Local Running Instructions

1. **Clone or save all 3 files (`app.py`, `requirements.txt`, `README.md`) into a single folder.**

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
