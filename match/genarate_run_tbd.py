import pandas as pd
import faiss
import numpy as np
import json
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm

# Uploading lucene model
searcher = LuceneSearcher('index/bm25_baseline_title')
searcher.set_bm25(k1=1.2, b=0.75)

df_examples = pd.read_parquet(
    r'data\shopping_queries_dataset\shopping_queries_dataset_examples.parquet',
    columns=['query_id', 'query', 'product_id', 'esci_label', 'product_locale', 'split', 'small_version']
)

df_queries = df_examples[
    (df_examples['product_locale'] == 'us') & 
    (df_examples['split'] == 'test') & 
    (df_examples['small_version'] == 1)
].copy()

queries_text = df_queries['query'].tolist()
queries_ids = df_queries['query_id'].tolist()

with open("runs/run_baseline_title.txt", 'w', encoding='utf-8') as f:
    for _, row in tqdm(df_queries.iterrows(), desc="Salvando dados da run_baseline_title", total=len(df_queries)):
        q_id = row['query_id']
        hits = searcher.search(str(row['query']), k=100)
        
        for rank, hit in enumerate(hits):
            f.write(f"{q_id} Q0 {hit.docid} {rank+1} {hit.score:.4f} BM25_Baseline\n")