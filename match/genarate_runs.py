import pandas as pd
import faiss
import numpy as np
import json
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm

# Uploading lucene model
searcher = LuceneSearcher('index/bm25_index_doc2query_qwen')
searcher.set_bm25(k1=1.2, b=0.75)

# Uploading sbert model all_MiniLM...
# model = SentenceTransformer('all-MiniLM-L6-v2')
# index = faiss.read_index("index/hnsw/amazon_hnsw.index")

# with open("index/hnsw/hnsw_mapping.json", 'r', encoding='utf-8') as f:
#     mapping = json.load(f)

df_queries = pd.read_csv("queries_poc_doc2query.csv")
df_queries_unique = df_queries[['query_id', 'query']].drop_duplicates()

with open("runs/doc2query/run_expanded_qwen.txt", 'w', encoding='utf-8') as f:
    for _, row in tqdm(df_queries_unique.iterrows(), total=len(df_queries_unique), desc="Gerando run baseline"):
        q_id = row['query_id']
        hits = searcher.search(str(row['query']), k=100)
        
        for rank, hit in enumerate(hits):
            f.write(f"{q_id} Q0 {hit.docid} {rank+1} {hit.score:.4f} BM25_Expanded_qwen\n")