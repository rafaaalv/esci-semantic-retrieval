import pandas as pd
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import CrossEncoder
from tqdm import tqdm

# Uploading lucene model
searcher = LuceneSearcher('index/bm25_baseline')
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


with open("run_baselineReRanked.txt", 'w', encoding='utf-8') as f:
    for _, row in df_queries.iterrows():
        q_id = row['query_id']
        hits = searcher.search(str(row['query']), k=100)
        
        for rank, hit in enumerate(hits):
            f.write(f"{q_id} Q0 {hit.docid} {rank+1} {hit.score:.4f} BM25_Baseline\n")

