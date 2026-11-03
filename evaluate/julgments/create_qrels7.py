import pandas as pd

# creating qrels file
# df_examples = pd.read_parquet(
#     r'data\shopping_queries_dataset\shopping_queries_dataset_examples.parquet',
#     columns=['query_id', 'query', 'product_id', 'esci_label', 'product_locale']
# )
# df_us = df_examples[df_examples['product_locale'] == 'us']
df_examples = pd.read_parquet(
    r'data\shopping_queries_dataset\shopping_queries_dataset_examples.parquet',
    columns=['query_id', 'query', 'product_id', 'esci_label', 'product_locale', 'split', 'small_version']
)

df_queries = df_examples[
    (df_examples['product_locale'] == 'us') &
    (df_examples['split'] == 'test') & 
    (df_examples['small_version'] == 1)
].copy()


qrels = df_queries[['query_id', 'product_id', 'esci_label']].copy()
print(len(df_queries))
qrels = qrels.rename(columns={'query_id': 'qid', 'product_id': 'docno'})
qrels['label'] = qrels['esci_label'].map({'E': 7, 'S': 3, 'C': 1, 'I': 0}).fillna(0)
qrels['qid'] = qrels['qid'].astype(str)
qrels['docno'] = qrels['docno'].astype(str)

with open("qrels7.txt", "w", encoding="utf-8") as f:
    for _, row in qrels.iterrows():
        f.write(f"{row['qid']} 0 {row['docno']} {row['label']}\n")