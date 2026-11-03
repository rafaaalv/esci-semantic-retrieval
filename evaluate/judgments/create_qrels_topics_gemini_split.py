import pandas as pd

print("1. Carregando os dados e resgatando as Chaves Primárias...")
# Lê o gabarito limpo que geramos anteriormente
df_gabarito = pd.read_csv("gabarito_poc_ndcg.csv")

# Lê o dataset original apenas para puxar os IDs verdadeiros (economiza RAM)
df_amazon_original = pd.read_parquet(
    "data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet", 
    columns=['query', 'query_id']
)

# Cria o dicionário de tradução { 'texto da busca' : 'ID_ORIGINAL' }
dicionario_ids = (
    df_amazon_original
    .drop_duplicates(subset=['query'])
    .set_index('query')['query_id']
    .to_dict()
)

# Injeta o ID oficial no nosso gabarito
df_gabarito['query_id'] = df_gabarito['query'].map(dicionario_ids)

# ==========================================
# 2. GERANDO O ARQUIVO TOPICS
# ==========================================
print("\n2. Gerando o arquivo TOPICS (topics_poc.tsv)...")
# Padrão TREC: query_id \t texto_da_query
df_topics = df_gabarito[['query_id', 'query']].drop_duplicates().sort_values('query_id')

df_topics.to_csv("topics_poc.tsv", sep='\t', index=False, header=False)
print(f" -> Sucesso! {len(df_topics)} consultas oficiais salvas.")

# ==========================================
# 3. GERANDO O ARQUIVO QRELS
# ==========================================
print("\n3. Gerando o arquivo QRELS (qrels_poc.txt)...")
# Padrão TREC: query_id \t 0 \t doc_id \t score
# E (Exact) = 3 | S (Substitute) = 2 | C (Complement) = 1 | I (Irrelevant) = 0
mapa_scores = {'E': 3, 'S': 2, 'C': 1, 'I': 0}
df_gabarito['score'] = df_gabarito['esci_label'].map(mapa_scores).fillna(0).astype(int)

# O "0" é uma exigência histórica de formatação do padrão TREC
df_gabarito['iteracao'] = "0"

# Monta a estrutura final rigorosa
df_qrels = df_gabarito[['query_id', 'iteracao', 'product_id', 'score']]

# Ordena por ID da query e pelas notas mais altas para manter o arquivo organizado
df_qrels = df_qrels.sort_values(by=['query_id', 'score'], ascending=[True, False])

df_qrels.to_csv("qrels_poc.txt", sep=' ', index=False, header=False)
print(f" -> Sucesso! {len(df_qrels)} julgamentos de relevância mapeados.")