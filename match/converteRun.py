import csv

input_file = "run_openAI_faiss_FINAL.csv" 
output_file = "run_openAI_faiss_FINAL.txt"


with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
    reader = csv.reader(f_in, delimiter=',')
    
    current_query = None
    current_rank = 1
    
    for row in reader:
        if not row or len(row) < 3:
            continue
            
        qid, docno, score = row[0], row[1], row[2]
        
        if qid == 'query_id':
            continue
            
        if qid != current_query:
            current_query = qid
            current_rank = 1

        trec_line = f"{qid} Q0 {docno} {current_rank} {score} Faiss_openAI\n"
        f_out.write(trec_line)
        
        current_rank += 1
