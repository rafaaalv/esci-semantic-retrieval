import pandas as pd
import json
import os
import math
from tqdm import tqdm
import pyarrow.parquet as pq

file_parquet = pq.ParquetFile("data\shopping_queries_dataset\shopping_queries_dataset_products.parquet")
file_out_jsonl = "batch_request.jsonl"
batch_size = 100000
rows = file_parquet.metadata.num_rows
chunks = math.ceil(rows/batch_size)

with open(file_out_jsonl, "w", encoding='utf-8') as f:
    for batch in tqdm(file_parquet.iter_batches(batch_size=batch_size), total=chunks):
        chunk = batch.to_pandas()
        chunk_us = chunk[chunk['product_locale'] == 'us'].dropna(subset=['product_id', 'product_title'])

        for _, row in chunk_us.iterrows():
            brand = str(row['product_brand']).strip() if pd.notna(row['product_brand']) else ""
            title = str(row["product_title"]).strip()
            if brand and brand.lower() != 'nan':
              text_to_embed = f"title: {title} | text: brand: {brand}"
            else:
              text_to_embed = f"title: {title} | text: none"
            req = {
                "key": str(row['product_id']),
                "request": {
                    "contents": [{"parts": [{"text": text_to_embed}]}]
                }
            }
            
            f.write(json.dumps(req) + '\n')