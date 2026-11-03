import pandas as pd
import json
import os

# run this command to index:
# py -m pyserini.index.lucene --collection JsonCollection --input amazon_lucene_collection_full --index index/bm25_full_baseline --generator DefaultLuceneDocumentGenerator --threads 4 --language en --storePositions --storeDocvectors --storeRaw

output_dir = "amazon_lucene_collection_full"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "documents.jsonl")

df_products = pd.read_parquet(
    r'data\shopping_queries_dataset\shopping_queries_dataset_products.parquet',
    columns=['product_id', 'product_title', 'product_brand', 'product_color', 'product_bullet_point', 'product_locale']
)

df_products = df_products.drop_duplicates(subset=['product_id'])
df_products['product_title'] = df_products['product_title'].fillna("")
df_products['product_brand'] = df_products['product_brand'].fillna("")
df_products['product_color'] = df_products['product_color'].fillna("")
df_products['product_bullet_point'] = df_products['product_bullet_point'].fillna("")
df_products = df_products[df_products['product_locale'] == 'us']

# Exporting to Pyserini format
with open(output_file, 'w', encoding='utf-8') as f:
    for index, row in df_products.iterrows():
        doc_id = str(row['product_id'])
        title = str(row['product_title']).strip()
        brand = str(row['product_brand']).strip()
        color = str(row['product_color']).strip()
        description = str(row['product_bullet_point']).strip()
            
        full_text = f"{title} {brand} {color} {description}"
        doc = {
            "id": doc_id,
            "contents": full_text
        }
        f.write(json.dumps(doc) + '\n')
