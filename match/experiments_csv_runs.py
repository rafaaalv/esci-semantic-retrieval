import pandas as pd
import pyterrier as pt
import gc

if not pt.started():
    pt.init()


runs_file = [
    ('run_baseline_title', "runs/run_baseline_title_novo.txt"),
    ('run_openAI_faiss_title', "runs/run_openAI_faiss_title.txt"),
    ('run_baseline_brand', "runs/run_baseline_brand.txt"),
    ('run_openAI_faiss_brand', "runs/run_openAI_faiss_brand.txt"),
    ('run_baseline_desc', "runs/run_baseline_description.txt"),
    ('run_openAI_faiss_desc', "runs/run_openAI_faiss_descp.txt"),
    ('run_openAI_faiss_title_label', "runs/run_openAI_faiss_title_label.txt"),
    ('run_openAI_faiss_brand_label', "runs/run_openAI_faiss_brand_label.txt"),
    ('run_openAI_faiss_desc_label', "runs/run_openAI_faiss_descp_label.txt"),
]



print("Lendo qrels")
qrels = pd.read_csv("qrels.txt", sep=" ", names=["qid", "iter", "docno", "label"], dtype=str)
qrels['label'] = qrels['label'].astype(int)

print("Lendo topics")
topics = pd.read_csv("topics_help.csv", sep=";", dtype=str)
topics = topics.rename(columns={'query_id': 'qid'})

print("Tamanho topics: " + str(len(topics)))
print("Tamanho qrels: " + str(len(qrels)))

df_ndcg10 = topics[['qid']].drop_duplicates().copy()
df_ndcg100 = topics[['qid']].drop_duplicates().copy()

for run, path in runs_file:
    print(f"Run atual {run}")
    run_df = pt.io.read_results(path)
    eval_df = pt.Experiment(
        retr_systems=[run_df],
        topics=topics,
        qrels=qrels,
        eval_metrics=["ndcg_cut_100"], 
        names=[run],
        perquery=True
    )

    m100 = eval_df[eval_df['measure'] == 'ndcg_cut_100'][['qid', 'value']].copy()
    m100 = m100.rename(columns={'value': run})
    df_ndcg100 = pd.merge(df_ndcg100, m100, on='qid', how='left')

    del run_df
    del eval_df
    del m100
    gc.collect()

df_ndcg100.to_csv("resultados_ndcg100_por_query.csv", index=False)
