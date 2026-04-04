import os
import requests
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_community.embeddings import HuggingFaceEmbeddings
import nltk
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

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



def query_llm(prompt):




    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.getenv("HF_TOKEN"),
    )

    completion = client.chat.completions.create(
        model="CohereLabs/tiny-aya-global:cohere",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )
    return completion.choices[0].message
def hybrid_retrieve(query, top_k=TOP_K, alpha=ALPHA):
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

if __name__ == "__main__":
    query_text = "Who played the hulk"
    print(f"\nPerforming hybrid retrieval for question: '{query_text}'")
    
    retrieved_docs = hybrid_retrieve(query_text)
    
    matches = retrieved_docs.get('matches', [])
    if not matches:
        print("No documents matched the query.")
    else:
        context = "\n".join([res['metadata']['text'] for res in retrieved_docs['matches'] if 'text' in res['metadata']])
        print(context)
        augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
        
        print("\nSending context and question to LLM (Tiny Aya Global)...")
        llm_response = query_llm(augmented_prompt)
        
        print("\n--- LLM Response ---")
        print(llm_response.content)
