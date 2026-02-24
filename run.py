from stable_baselines3 import PPO
from rag import SimpleRAG
from features import extract_features
from llm_api import call_llm

rag = SimpleRAG("document.txt")

model = PPO.load("rl_guard_policy")

def ask(query):
    context = rag.retrieve(query)

    prompt = f"Context:\n{context}\n\nQuestion:\n{query}\nAnswer:"
    raw_answer = call_llm(prompt)

    state = extract_features(raw_answer, context, query)
    print("STATE:", state)
    action, _ = model.predict(state)

    if action == 1:
        final_answer = call_llm(prompt, temperature=0.4)
    elif action == 2:
        final_answer = "Answer rejected by RL layer."
    else:
        final_answer = raw_answer

    print("\n==============================")
    print("RAW OPENAI ANSWER:\n", raw_answer)
    print("\nRL FILTERED ANSWER:\n", final_answer)
    print("==============================\n")

while True:
    q = input("Ask: ")
    ask(q)