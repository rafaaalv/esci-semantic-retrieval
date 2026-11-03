import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import time
import os

client = OpenAI("API-KEY")

file_out = "indexing_openAI_esci_products_title.csv"

df_products = pd.read_parquet(
    r'data\shopping_queries_dataset\shopping_queries_dataset_products.parquet',
    columns=['product_id', 'product_title', 'product_locale']
)
df_products = df_products.drop_duplicates(subset=['product_id'])
df_products = df_products[df_products['product_locale'] == 'us']
produtos_anteriores = set()
if os.path.exists(file_out):
    try:
        df_salvo = pd.read_csv(file_out, usecols=['product_id'])
        produtos_anteriores = set(df_salvo['product_id'])
    except Exception:
        print("   -> Arquivo vazio ou corrompido, o processo iniciará do zero.")

df_pendentes = df_products[~df_products['product_id'].isin(produtos_anteriores)].copy()

if df_pendentes.empty:
    print("\nTodos os produtos já foram vetorizados! O arquivo está completo.")
    exit()

df_pendentes['product_title'] = df_pendentes['product_title'].fillna("")
df_pendentes['text_to_embed'] = df_pendentes['product_title']

text = df_pendentes['text_to_embed'].tolist()
batch_size = 500

for i in tqdm(range(0, len(text), batch_size), desc="Indexando vetores OpenAI"):
    batch = text[i:i + batch_size]
    max_retries = 3
    
    batch_vectors = []
    
    for t in range(max_retries):
        try: 
            response = client.embeddings.create(input=batch, model="text-embedding-3-small")
            batch_vectors = [d.embedding for d in response.data]
            break
        except Exception as e:
            print(f"\nErro na API: {e}. Tentando novamente...")
            time.sleep(5)
            if t == max_retries - 1:
                batch_vectors = [None] * len(batch)
    df_lote = df_pendentes.iloc[i:i + batch_size].copy()
    df_lote['embedding'] = batch_vectors
    df_lote = df_lote.dropna(subset=['embedding'])
    
    df_lote_final = df_lote[['product_id', 'text_to_embed', 'embedding']]
    
    df_lote_final.to_csv(file_out, mode='a', header=not os.path.exists(file_out), index=False)