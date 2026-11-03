import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import time
import os

client = OpenAI("API-KEY")

file_out = "indexing_openAI_esci_queries.csv"

df_examples = pd.read_parquet(
    r'data\shopping_queries_dataset\shopping_queries_dataset_examples.parquet',
    columns=['query_id', 'query', 'product_id', 'esci_label', 'product_locale', 'split', 'small_version']
)
df_examples = df_examples.drop_duplicates(subset=['query_id']).copy()
df_queries = df_examples[
    (df_examples['product_locale'] == 'us') & 
    (df_examples['split'] == 'test') & 
    (df_examples['small_version'] == 1)
].copy()

print(f"Tamanho do arquivo de queries {len(df_queries)}")

queries_anteriores = set()

if os.path.exists(file_out):
    try:
        df_salvo = pd.read_csv(file_out, usecols=['query_id'])
        queries_anteriores = set(df_salvo['query_id'])
    except Exception:
        print("   -> Arquivo vazio ou corrompido, o processo iniciará do zero.")

df_pendentes = df_queries[~df_queries['query_id'].isin(queries_anteriores)].copy()

if df_pendentes.empty:
    print("\nTodos as queries já foram vetorizadas! O arquivo está completo.")
    exit()

text = df_pendentes['query'].tolist()
batch_size = 500

for i in tqdm(range(0, len(text), batch_size)):
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
    
    df_lote_final = df_lote[['query_id', 'query', 'embedding']]
    
    df_lote_final.to_csv(file_out, mode='a', header=not os.path.exists(file_out), index=False)