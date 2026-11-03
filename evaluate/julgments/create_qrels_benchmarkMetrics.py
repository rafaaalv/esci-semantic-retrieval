import pandas as pd

# creating qrels file
df_examples = pd.read_parquet(
    r'data\shopping_queries_dataset\shopping_queries_dataset_examples.parquet',
    columns=['query_id', 'product_id', 'esci_label', 'product_locale', 'split', 'small_version']
)
df_qrels = df_examples[
    (df_examples['product_locale'] == 'us') &
    (df_examples['split'] == 'test') &
    (df_examples['small_version'] == 1)
].copy()

print(f"Total de buscas no gabarito final: {len(df_qrels)} linhas.")

qrels = df_qrels[['query_id', 'product_id', 'esci_label']].copy()
qrels = qrels.rename(columns={'query_id': 'qid', 'product_id': 'docno'})
qrels['label'] = qrels['esci_label'].map({'E': 100, 'S': 10, 'C': 1, 'I': 0}).fillna(0)
qrels['qid'] = qrels['qid'].astype(str)
qrels['docno'] = qrels['docno'].astype(str)

with open("qrels_benchmarkMetrics.txt", "w", encoding="utf-8") as f:
    for _, row in qrels.iterrows():
        f.write(f"{row['qid']} 0 {row['docno']} {row['label']}\n")