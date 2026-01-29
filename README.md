# 🧠 Last Minute Prep – AI Study Assistant

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![LangChain](https://img.shields.io/badge/LangChain-LLM-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**AI-powered study assistant that converts PDFs and YouTube videos into cheat sheets, quizzes, and exam-ready notes using LLMs and vector search.**

</div>

---

## 🎯 Overview

**Last Minute Prep** is a multi-tool **Streamlit application** designed for efficient exam preparation.  
It processes PDFs and YouTube videos using **adaptive LLM strategies** to generate concise, accurate, and comprehensive study material while respecting token limits.

**Key Idea:** Automatically switches between **Direct / Hybrid / MapReduce** strategies based on content size.

---

## ✨ Features

### 📚 PDF Study Assistant
- Cheat sheet generation (formulas, key points, concepts)
- Interactive quizzes with instant feedback
- Semantic **PDF Q&A using FAISS**
- Memory aids, mnemonics & exam question prediction
- Adaptive processing for large documents (up to ~50 pages)

### 🎥 YouTube Summarizer
- MapReduce-based long video summarization
- Transcript-based summaries (no hallucinations)
- Hindi → English auto-translation
- Multiple summary styles (concise, detailed, bullet points)

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-----|-------------|
| LLM | Groq (LLaMA-3.3-70B), LangChain |
| Vector DB | FAISS |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Frontend | Streamlit, Custom CSS |
| Docs | PyPDF2, ReportLab, Markdown2 |
| APIs | YouTube Transcript API |

---

## 📁 Project Structure

```

├── app_final.py          # Main router & landing page
├── cheatsheet_app.py     # PDF processing + cheat sheets
├── pdf_qa_app.py         # Semantic PDF Q&A (FAISS)
├── yt_summary_app.py     # YouTube summarizer
├── requirements.txt
├── .env                 # API keys (not committed)
└── .streamlit/config.toml

````

---

## 🧠 Core Technical Highlights

### 🔹 Adaptive Processing
```python
if len(content) <= 15000:
    strategy = "direct"
elif len(content) <= 35000:
    strategy = "hybrid"
else:
    strategy = "mapreduce"
````

### 🔹 FAISS-based Semantic Search

```python
vector_store = FAISS.from_texts(chunks, embeddings)
relevant_chunks = vector_store.similarity_search(query, k=4)
```

### 🔹 MapReduce for Long Content

```python
map_outputs = [llm(chunk) for chunk in chunks]
final_output = llm.combine(map_outputs)
```

### 🔹 Safe LLM Calls

```python
def safe_llm_call(model, prompt):
    try:
        return model.invoke(prompt).content
    except RateLimitError:
        return None
```

---

## 🚀 Installation

```bash
git clone https://github.com/akshat657/last-minute-prep.git
cd last-minute-prep
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env
streamlit run app_final.py
```

Free Groq API key: [https://console.groq.com](https://console.groq.com)

---

## 📊 Performance Snapshot

| Feature              | Time | Tokens |
| -------------------- | ---- | ------ |
| Small PDF (15 pages) | ~10s | ~3K    |
| Large PDF (50 pages) | ~60s | ~15K   |
| PDF Q&A              | ~5s  | ~2K    |
| YouTube Summary      | ~20s | ~8K    |

---

## 🚀 Deployment

**Streamlit Cloud (Recommended)**

1. Push repo to GitHub
2. Connect at [https://share.streamlit.io](https://share.streamlit.io)
3. Add `GROQ_API_KEY` in secrets
4. Deploy

*(Free tier apps may sleep after inactivity.)*

---

## 🤝 Contributing

Ideas welcome:

* OCR for scanned PDFs
* Flashcard generation
* More LLM providers
* Multi-document comparison

---

## 📄 License

MIT License

---

## 👨‍💻 Author

**Akshat Khandelwal**
GitHub: [https://github.com/akshat657](https://github.com/akshat657)
LinkedIn: [https://linkedin.com/in/akshat-khandelwal](https://linkedin.com/in/akshat-khandelwal)
Email: [akshatkhandelwal004@gmail.com](mailto:akshatkhandelwal004@gmail.com)

---

<div align="center">

**Built with Python, Streamlit, LangChain, and FAISS**
*Making last-minute exam prep smarter, not harder.*

</div>
