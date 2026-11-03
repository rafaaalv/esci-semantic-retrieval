import pandas as pd
df_examples = pd.read_parquet(
    r'data\shopping_queries_dataset\shopping_queries_dataset_examples.parquet',
    columns=['query_id', 'query', 'product_id', 'esci_label', 'product_locale', 'split', 'small_version']
)

df_queries = df_examples[
    (df_examples['product_locale'] == 'us') &
    (df_examples['split'] == 'test') & 
    (df_examples['small_version'] == 1)
].copy()


print(len(df_queries))
df_topics = df_queries[['query_id', 'product_id', 'esci_label']].drop_duplicates()
df_topics.to_csv("topics_help.csv", index=False, sep=";")