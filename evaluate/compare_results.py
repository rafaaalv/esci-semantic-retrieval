import pandas as pd
import pyterrier as pt

if not pt.started():
    pt.init()

print("Lendo runs sem label")
run_baseline_title = pt.io.read_results("runs/run_baseline_title_novo.txt")
run_baseline_brand = pt.io.read_results("runs/run_baseline_brand.txt")
run_baseline_desc = pt.io.read_results("runs/run_baseline_description.txt")



print("Lendo qrels")
qrels = pd.read_csv("qrels.txt", sep=" ", names=["qid", "iter", "docno", "label"], dtype=str)
qrels['label'] = qrels['label'].astype(int)

print("Lendo topics")
topics = pd.read_csv("topics_help.csv", sep=";", dtype=str)
topics = topics.rename(columns={'query_id': 'qid'})

print("Tamanho baseline title: " + str(len(run_baseline_title)))
print("Tamanho baseline brand: " + str(len(run_baseline_brand)))
print("Tamanho baseline desc: " + str(len(run_baseline_desc)))
print("Tamanho topics: " + str(len(topics)))
print("Tamanho qrels: " + str(len(qrels)))

print("Executando exeperimento...")
experiment1 = pt.Experiment(
    retr_systems=[run_baseline_title, run_baseline_brand, run_baseline_desc],
    topics=topics,
    qrels=qrels,
    eval_metrics=["ndcg_cut_10", "ndcg_cut_100"], 
    names=["1_BM25_Title", "2_BM25_Brand", "#_BM25_Description"],
    baseline = 0,
)
print("===== RESULTS ======\n")
print(experiment1)
print("====================\n")
print("Salvando csv")
experiment1.to_csv("Final_Result_bm25OpenAI_title.csv", index=False, sep=';', decimal='.')