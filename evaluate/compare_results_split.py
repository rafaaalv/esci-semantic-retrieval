import pandas as pd
import pyterrier as pt

if not pt.started():
    pt.init()

print("Lendo runs")
run_bm25_baseline = pt.io.read_results("runs/doc2query/run_baseline.txt")
run_bm25_expanded = pt.io.read_results("runs/doc2query/run_expanded_5_4.txt")
run_bm25_expanded_nano = pt.io.read_results("runs/doc2query/run_expanded_5_4_nano.txt")
run_bm25_expanded_4 = pt.io.read_results("runs/doc2query/run_expanded_4o_mini.txt")
run_bm25_expanded_qwen = pt.io.read_results("runs/doc2query/run_expanded_qwen.txt")

df_queries = pd.read_csv("queries_poc_doc2query.csv")

print("Lendo qrels")
qrels = pd.read_csv("qrels_poc.txt", sep=" ", names=["qid", "iter", "docno", "label"], dtype=str)

print("Lendo topics")
df_topics = pd.read_csv("queries_poc_doc2query.csv")
df_topics = df_topics.rename(columns={
    'query_id': 'qid'
})
df_topics['qid'] = df_topics['qid'].astype(str)
df_topics['query'] = df_topics['query'].astype(str)
qrels['label'] = qrels['label'].astype(int)
qrels['qid'] = qrels['qid'].astype(str)
qrels['docno'] = qrels['docno'].astype(str)
print("Tamanho topics: " + str(len(df_topics)))
print("Tamanho qrels: " + str(len(qrels)))

print("Executando o experimento")
experiment = pt.Experiment(
    retr_systems=[run_bm25_baseline, run_bm25_expanded, run_bm25_expanded_nano, run_bm25_expanded_4, run_bm25_expanded_qwen],
    topics=df_topics,
    qrels=qrels,
    eval_metrics=["ndcg_cut_10", "ndcg_cut_100"], 
    names=["bm25_baseline", "bm25_doc2query", "bm25_doc2query_nano", "bm25_doc2query_4", "bm25_doc2query_qwen"],
)

print(experiment)