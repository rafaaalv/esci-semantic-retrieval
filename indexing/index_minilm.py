import pandas as pd
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Uploading sbert model
model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384 

df_products = pd.read_parquet(
    r'data\shopping_queries_dataset\shopping_queries_dataset_products.parquet',
    columns=['product_id', 'product_title', 'product_locale']
)
df_products = df_products[df_products['product_locale'] == 'us'].drop_duplicates(subset=['product_id'])
df_products['product_title'] = df_products['product_title'].fillna("")

products_list = df_products.to_dict('records')
embedding_text = [p['product_title'] for p in produtos_lista]

index = faiss.IndexHNSWFlat(dimension, 32)
index.hnsw.efSearch = 64 

embeddings = model.encode(embedding_text, batch_size=256, show_progress_bar=True)

embeddings = np.array(embeddings).astype('float32')
index.add(embeddings)

os.makedirs("index/hnsw", exist_ok=True)

faiss.write_index(index, "index/hnsw/amazon_hnsw.index")

ids_mapping = {i: p['product_id'] for i, p in enumerate(products_list)}
with open("index/hnsw/hnsw_mapping.json", 'w', encoding='utf-8') as f:
    json.dump(ids_mapping, f)
