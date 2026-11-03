import pandas as pd
import json
import os
from tqdm import tqdm

# run this command to index:
# py -m pyserini.index.lucene --collection JsonCollection --input amazon_lucene_collection_tbd --index index/bm25_baseline_tbd --generator DefaultLuceneDocumentGenerator --threads 4 --language en --storePositions --storeDocvectors --storeRaw

output_dir = "amazon_lucene_collection_tbd"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "documents.jsonl")

df_products = pd.read_parquet(
    r'data\shopping_queries_dataset\shopping_queries_dataset_products.parquet',
    columns=['product_id', 'product_title', 'product_locale', 'product_description', 'product_brand']
)

df_products = df_products.drop_duplicates(subset=['product_id'])
df_products['product_title'] = df_products['product_title'].fillna("")
df_products['product_description'] = df_products['product_description'].fillna("")
df_products['product_brand'] = df_products['product_brand'].fillna("")
df_products = df_products[df_products['product_locale'] == 'us']

# Exporting to Pyserini format
with open(output_file, 'w', encoding='utf-8') as f:
    for index, row in tqdm(df_products.iterrows(), desc="Indexando title brand description", total=len(df_products)):
        doc_id = str(row['product_id'])
        title = str(row['product_title']).strip()
        brand = str(row['product_brand']).strip()
        description = str(row['product_description']).strip()

        content = title + " " + brand + " " + description
        doc = {
            "id": doc_id,
            "contents": content
        }
        f.write(json.dumps(doc) + '\n')