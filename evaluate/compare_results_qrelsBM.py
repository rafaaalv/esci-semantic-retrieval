import pandas as pd
import pyterrier as pt

if not pt.started():
    pt.init()

run_baseline = pt.io.read_results("run_baseline.txt")
run_hnsw = pt.io.read_results("run_faiss.txt")

qrels = pd.read_csv("qrels_benchmarkMetrics.txt", sep=" ", names=["qid", "iter", "docno", "label"], dtype=str)
qrels['label'] = qrels['label'].astype(int)

topics = pd.read_csv("topics_help.csv", sep=";", dtype=str)
topics = topics.rename(columns={'query_id': 'qid'})
# print(len(topics))

experiment = pt.Experiment(
    retr_systems=[run_baseline, run_hnsw],
    topics=topics,
    qrels=qrels,
    eval_metrics=["ndcg_cut_100"], 
    names=["1_BM25_Lexical", "2_Faiss_HNSW"],
    baseline = 0
)
print("===== RESULTS ======\n")
print(experiment)
print("====================\n")

experiment.to_csv("Final_Result_BM.csv", index=False, sep=';', decimal='.')
