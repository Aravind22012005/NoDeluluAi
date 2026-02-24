from stable_baselines3 import PPO
from rag import SimpleRAG
from env import RAGGuardEnv

rag = SimpleRAG("document.txt")

queries = [
    "Summarize the document",
    "What are the key points?",
    "Explain the main idea",
]

env = RAGGuardEnv(rag, queries)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=16,
    batch_size=16
)

model.learn(total_timesteps=1000)

model.save("rl_guard_policy")