import os
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

def judge_llm(prompt, model="qwen/qwen3.5-flash-02-23"): #default to model of choice else passed different for experiments
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OR_TOKEN"),
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=512
    )
    return completion.choices[0].message

def faithfulness(gen_answer, context):
    extraction_prompt = f"Extract factual claims from this answer as a list. Only respond with the list. \n Answer: {gen_answer}\n Claims:"
    claims_raw = judge_llm(extraction_prompt).content
    #print(claims_raw)
    claims = [c.strip("- ") for c in claims_raw.split("\n") if c.strip()]
    supported=0
    details = []
    for claim in claims:
        verify=f"Context: {context}\n Claim:{claim}\n Is this claim supported by the context? Answer 'Yes' or 'No' only."
        verdict = judge_llm(verify).content
        if "Yes" in verdict: supported += 1
        details.append({"claim": claim, "verdict": verdict})
    score = supported/len(claims) if claims else 0
    return score, details

def answer_relevancy(gen_answer, query, embed_model):
    prompt = f"Generate 3 questions this answer addresses.Respond with the questions only. \n Answer: {gen_answer} Questions:"
    gen_qs = judge_llm(prompt).content.split('\n')[:3]
    #print(gen_qs)
    orig_vec = np.array(embed_model.embed_query(query)).reshape(1, -1)
    similarities = []
    for q in gen_qs:
        gen_vec = np.array(embed_model.embed_query(q)).reshape(1, -1)
        similarities.append(cosine_similarity(orig_vec, gen_vec)[0][0]) #comp cosine
    return np.mean(similarities)

