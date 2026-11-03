import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import os

client = OpenAI(api_key="")

arquivo_parquet = "produtos_poc_doc2query.parquet"
arquivo_saida = "produtos_poc_expandido_4o.jsonl"

df_produtos = pd.read_parquet(arquivo_parquet).fillna("")

produtos_processados = set()
if os.path.exists(arquivo_saida):
    with open(arquivo_saida, 'r', encoding='utf-8') as f:
        for linha in f:
            if linha.strip():
                try:
                    dado = json.loads(linha)
                    pid = str(dado.get('id_garantido', ''))
                    if pid:
                        produtos_processados.add(pid)
                except json.JSONDecodeError:
                    continue

coluna_id = 'product_id' if 'product_id' in df_produtos.columns else 'id'
df_restante = df_produtos[~df_produtos[coluna_id].astype(str).isin(produtos_processados)]

print(f"Retomando de onde parou. Faltam processar: {len(df_restante)} produtos.")

system_prompt = """You are an expert e-commerce search optimization system.
Your task is to read the product title and description, and generate exactly 5 distinct search queries a real customer would type to find this product. 
Include:
- 1 broad category query
- 2 feature-specific queries
- 2 synonym or problem-solving queries

Format rules: 
- Return ONLY the queries separated by commas.
- Do not use quotes, bullet points, or conversational text."""

with open(arquivo_saida, 'a', encoding='utf-8') as f:
    
    for index, row in tqdm(df_restante.iterrows(), total=df_restante.shape[0], desc="Doc2Query API"):
        
        pid = str(row.get(coluna_id, index))
        titulo = str(row.get('product_title', row.get('title', '')))
        descricao = str(row.get('product_description', row.get('description', '')))
        
        user_prompt = f"Title: {titulo}\nDescription: {descricao}"

        try:
            resposta = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2, 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            termos_gerados = resposta.choices[0].message.content.strip().replace(",", "")
        
        except Exception as e:
            print(f"\n[ERRO API] Timeout ou queda na rede no produto {pid}. Salvando vazio e avançando...")
            termos_gerados = ""

        texto_expandido = f"{titulo} {descricao} {termos_gerados}".strip()
        
        produto_expandido = row.to_dict()
        produto_expandido['id_garantido'] = pid
        produto_expandido['titulo_garantido'] = titulo
        
        produto_expandido['doc2query_expansion'] = termos_gerados
        produto_expandido['texto_expandido'] = texto_expandido
        
        linha_json = json.dumps(produto_expandido, ensure_ascii=False)
        
        f.write(linha_json + '\n')
        f.flush() 
        
print("\nProcessamento finalizado com total segurança de dados!")