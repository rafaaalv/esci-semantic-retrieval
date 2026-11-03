import pandas as pd
import json
import os
from tqdm import tqdm

# run this command to index:
# py -m pyserini.index.lucene --collection JsonCollection --input bm25_index_doc2query_qwen --index index/bm25_index_doc2query_qwen --generator DefaultLuceneDocumentGenerator --threads 4 --language en --storePositions --storeDocvectors --storeRaw

output_dir = "bm25_index_doc2query_qwen"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "documents.jsonl")

df_products = pd.read_json('produtos_poc_expandido_qwen.jsonl', lines=True)

# Exporting to Pyserini format
with open(output_file, 'w', encoding='utf-8') as f:
    for index, row in tqdm(df_products.iterrows(), total=len(df_products)):
        doc_id = str(row['product_id'])
        title = str(row['product_title']).strip()
        brand = str(row['product_brand']).strip()
        descp = str(row['product_description']).strip()
        expansion = str(row['doc2query_expansion']).strip()
            
        doc = {
            "id": doc_id,
            "contents": title + " " + brand + " " + descp + " " + expansion
        }
        f.write(json.dumps(doc) + '\n')
