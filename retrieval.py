import os
import requests
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_community.embeddings import HuggingFaceEmbeddings
import nltk
import time
from openai import OpenAI
from dotenv import load_dotenv
from judgellm import faithfulness, answer_relevancy
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

#GLOBAL VARIABLES
TYPE = "semantic" # semantic/recurive chunks
TOP_K = 6 # retrieved chunks
ALPHA = 0.7 # alpha for hybrid score


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

import pickle

# 3. Load existing BM25 encoder from pickle file
print("Loading existing bm25_encoder.pkl...")
with open("bm25_encoder.pkl", "rb") as f:
    bm25 = pickle.load(f)


def query_llm(prompt, model="CohereLabs/tiny-aya-global:cohere"): #default to model of choice else passed different for experiments
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.getenv("HF_TOKEN"),
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )
    return completion.choices[0].message

def hybrid_retrieve(query, embed_model, bm25, p_index, top_k=TOP_K, alpha=ALPHA):
    # Create dense and sparse vectors as usual
    dense_vec = embed_model.embed_query(query)
    sparse_vec = bm25.encode_queries(query)
    
    # Scale them
    scaled_dense = [v * alpha for v in dense_vec]
    scaled_sparse = {
        'indices': sparse_vec['indices'],
        'values': [v * (1 - alpha) for v in sparse_vec['values']]
    }

    # Query Pinecone with the TYPE FILTER
    result = p_index.query(
        vector=scaled_dense,
        sparse_vector=scaled_sparse,
        top_k=top_k,
        include_metadata=True,
        # THIS IS THE KEY PART:
        filter={
            "type": {"$eq": TYPE}
        }
    )
    return result

def reranker(query, retreived_docs, top_n=6):
    doc_texts = [res['metadata']['text'] for res in retrieved_docs['matches']]
    if not doc_texts:
        return []

    actual_top_n = min(len(doc_texts), top_n) #if 6 aren't returned
    api_url = f"https://api-inference.huggingface.co/models/{RERANK_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": {"source_sentence": query, "sentences": doc_texts}
    }
    response = requests.post(api_url, headers=headers, json=payload)
    
    if response.status_code == 200:
        scores = response.json()
        scored_docs = sorted(zip(doc_texts, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:actual_top_n]]
    else:
        return doc_texts[:actual_top_n]

def get_rag_response(query_text, chunk_type="semantic", alpha=0.7, use_rerank=True, model="CohereLabs/tiny-aya-global:cohere"):
    global TYPE
    TYPE = chunk_type #pinecone update
    start_total = time.perf_counter() #total time started
    start_retrieval = time.perf_counter()
    top_k_fetch = 20 if use_rerank else 6
    retrieved_docs = hybrid_retrieve(query_text, embed_model=embed_model, bm25=bm25, p_index=p_index, top_k=top_k_fetch, alpha=alpha)
    if use_rerank:
        final_chunks = reranker(query_text, retrieved_docs, top_n=6)
    else:
        final_chunks = [res['metadata']['text'] for res in retrieved_docs.get('matches', []) if 'text' in res.get('metadata', {})][:6]
    retrieval_time = time.perf_counter() - start_retrieval
    
    if not final_chunks:
        return "No relevant context found.", ""
    context = "\n".join(final_chunks)
    augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
    start_inference = time.perf_counter()
    llm_response = query_llm(augmented_prompt, model=model)
    inference_time = time.perf_counter() - start_inference
    total_time = time.perf_counter() - start_total
    metrics = {
        "retrieval_time": retrieval_time,
        "inference_time": inference_time,
        "total_time": total_time
    }
    return llm_response.content, context, metrics#setup for eval


if __name__ == "__main__": #main for testing
    query_text = "Who played the hulk"
    print(f"\nPerforming hybrid retrieval for question: '{query_text}'")
    
    retrieved_docs = hybrid_retrieve(query_text, embed_model=embed_model, bm25=bm25, p_index=p_index)
    print(f"\nReranking retreived documents: '{query_text}'")
    final_chunks = reranker(query_text, retrieved_docs, top_n=6)
    if not final_chunks:
        print("No documents matched the query.")
    else:
        context = "\n".join(final_chunks)
        print(context)
        augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
        
        print("\nSending context and question to LLM (Tiny Aya Global)...")
        llm_response = query_llm(augmented_prompt)
        
        print("\n--- LLM Response ---")
        print(llm_response.content)

        score, details = faithfulness(llm_response.content, context)
        relevancy = answer_relevancy(llm_response.content, query_text, embed_model)
        print("\n------Evaluating Response------")
        print(f"\nFaithfullness: {score}\n {details}\n Relevancy: {relevancy}")
