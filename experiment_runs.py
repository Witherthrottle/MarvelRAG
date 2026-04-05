import pandas as pd
import time
from judgellm import faithfulness, answer_relevancy
from retrieval import get_rag_response
import os
import requests
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_community.embeddings import HuggingFaceEmbeddings
import nltk
import time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# 1. Initialize Pinecone
print("Initializing Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Get the first available index
indexes = pc.list_indexes().names()
if not indexes:
    raise ValueError("No Pinecone indexes found!")
index_name = indexes[0]
print(f"Using Pinecone index: {index_name}")

p_index = pc.Index(index_name)

# 2. Setup embed model exactly as the vector documents
print("Loading HuggingFace Embeddings model...")
embed_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}, 
    encode_kwargs={
        "batch_size": 64,
        "normalize_embeddings": True
    }
)

def run_eval_batch(queries, config_label, chunk, alpha, rerank, model):
    results = []
    print(f"\n{'='*60}")
    print(f"  LLM TARGET: {config_label}")
    print(f"{'='*60}")

    for i, q in enumerate(queries, 1):
        try:
            ans, ctx, metrics = get_rag_response(q, chunk, alpha, rerank, model)
            
            # Run Evaluation Judges
            f_score, _ = faithfulness(ans, ctx)
            r_score = answer_relevancy(ans, q, embed_model)
            
            # Print real-time query results
            print(f"\n[Query {i}]: {q}")
            print(f" > Faithfulness: {f_score:.2f}")
            print(f" > Relevancy:    {r_score:.2f}")
            print(f"{'-'*30}")

            results.append({
                "query": q,
                "f_score": f_score,
                "r_score": r_score,
            })
            time.sleep(0.5) 
            
        except Exception as e:
            print(f"Error on {q}: {e}")
            
    df = pd.DataFrame(results)
    return {
        "Strategy Label": config_label,
        "Mean Faithfulness": df["f_score"].mean(),
        "Faithfulness Std": df["f_score"].std(),
        "Min Faithfulness Score": df["f_score"].min(),
        "Mean Relevancy": df["r_score"].mean(),
        "Relevancy Std:": df["r_score"].std(),
        "Min Relevancy Score" :df["r_score"].min()
    }

if __name__ == "__main__":
    test_queries = ["Which actors have played Spider-Man?",
                   "How did Iron Man die?",
                   "Who are the Avengers?",
                   "Who created Vision?",
                   "How did Thanos acquire the Soul Stone?",
                   "What is the Tesseract?",
                   "Who is the leader of S.H.I.E.L.D?",
                   "Where is Wakanda located?",
                   "Which actor played the Hulk?",
                  "How does Ant-Man shrink?",                  
 "Explain the evolution of Tony Stark's motivation from the first Iron Man to his sacrifice in Endgame.", 
 "My friend told me that iron man is better than captain america. do you think that is the case because i really like captian america"] #11 queries
    #test_queries = ["My friend told me that iron man is better than captain america. do you think that is the case because i really like captian america"]
    # # --- STAGE 1: LLM SELECTION ---
    # # Fixed: Alpha=0.5, Chunk=Recursive, Rerank=False
    # print("\n--- STAGE 1: Comparing LLM Models ---")
    # stage1_results = []
    # llms = [ 
    #     "meta-llama/llama-3-8b-instruct", 
    #     "mistralai/mistral-7b-instruct-v0.1"
    # ]
    # for m in llms:
    #     res = run_eval_batch(test_queries, m.split('/')[-1], "recursive", 0.5, False, m)
    #     stage1_results.append(res)
        
    
    # print(pd.DataFrame(stage1_results).to_string(index=False))

    # --- STAGE 2: RETRIEVAL STRATEGY (ALPHA) ---
    # Fixed: BEST_MODEL (from Stage 1), Chunk=Recursive, Rerank=False
    # print("\n--- STAGE 2: Comparing Alpha (Retrieval Strategy) ---")
    best_model = "mistralai/mistral-7b-instruct-v0.1" # Manually updated based on S1 results
    # stage2_results = []
    # alphas = { "Keyword-Biased Hybrid": 0.3, "Keyword Only":0.0, "Hybrid":0.5} #"Semantic Only": 1.0, "Skewed Hybrid": 0.7}
    
    # for label, a_val in alphas.items():
    #     res = run_eval_batch(test_queries, label, "recursive", a_val, False, best_model)
    #     stage2_results.append(res)
    
    # print(pd.DataFrame(stage2_results).to_string(index=False))

    # --- STAGE 3: CHUNKING & RERANKER (FINAL POLISH) ---
    # Fixed: BEST_MODEL, BEST_ALPHA (from Stage 2)
    print("\n--- STAGE 3: Comparing Chunking & Reranking ---")
    best_alpha = 0.0 
    stage3_results = []
    final_configs = [
        {"label": "Semantic + Rerank", "chunk": "semantic", "rerank": True},
        {"label": "Recursive + Rerank", "chunk": "recursive", "rerank": True},
    ]

    for cfg in final_configs:
        res = run_eval_batch(test_queries, cfg["label"], cfg["chunk"], best_alpha, cfg["rerank"], best_model)
        stage3_results.append(res)

    print(pd.DataFrame(stage3_results).to_string(index=False))