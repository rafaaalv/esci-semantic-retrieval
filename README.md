# E-commerce Semantic Search & LLM Query Expansion 

This repository contains a research-oriented Information Retrieval (IR) pipeline developed to evaluate search relevance methodologies using the **public Amazon ESCI (Shopping Queries Dataset)**. 

> **Note:** This project focuses strictly on academic/research applications of search architecture using publicly available datasets and open-source frameworks. It demonstrates the technical foundation of dense retrieval and Generative AI integration without containing proprietary business logic.

## 🚀 Key Features & Architecture
* **Data Ingestion:** Automated scripts to download, verify, and split the ESCI dataset.
* **Baseline Retrieval:** Implementation of classical lexical indexing (BM25).
* **Vector Search:** Dense retrieval utilizing HNSW with FAISS for fast approximate nearest neighbor search.
* **Generative AI Integration (Doc2Query):** Leveraging OpenAI and Google Gemini APIs to generate synthetic queries, expanding document representations before indexing to improve semantic match.
* **Evaluation Pipeline:** Automated scripts to generate relevance judgments (`qrels`), compute metrics, and compare baseline vs. LLM-augmented indexing strategies.

## 📁 Repository Structure
* `/data`: Dataset ingestion, verification, and train/test splitting scripts.
* `/doc2query`: Scripts for API consumption and automated document expansion using LLMs.
* `/indexing`: Setup for traditional and dense vector indexing.
* `/match`: Retrieval and semantic matching execution.
* `/evaluate`: Pipeline for relevance judgments generation and results comparison.

## 🛠️ Tech Stack
* **Language:** Python
* **IR Frameworks:** FAISS, PyTerrier, Pyserini
* **Models:** Hugging Face (all-MiniLM)
* **LLM APIs:** OpenAI API, Google Gemini API

## ⚙️ Setup & Execution

**1. Clone the repository and install dependencies:**
```bash
git clone [https://github.com/your-username/esci-semantic-retrieval.git](https://github.com/your-username/esci-semantic-retrieval.git)
cd esci-semantic-retrieval
pip install -r requirements.txt
