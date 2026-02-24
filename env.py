import gymnasium as gym
from gymnasium import spaces
import numpy as np
from features import extract_features
from llm_api import call_llm

class RAGGuardEnv(gym.Env):
    def __init__(self, rag, queries):
        super(RAGGuardEnv, self).__init__()

        self.rag = rag
        self.queries = queries

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=0,
            high=1000,
            shape=(4,),
            dtype=np.float32
        )

        self.max_steps = 3

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.query = np.random.choice(self.queries)

        self.context = self.rag.retrieve(self.query)

        prompt = f"Context:\n{self.context}\n\nQuestion:\n{self.query}\nAnswer:"
        self.answer = call_llm(prompt)

        self.state = extract_features(self.answer, self.context, self.query)

        return self.state, {}

    def step(self, action):
        done = False
        self.step_count += 1

        if action == 1:  # regenerate
            prompt = f"Context:\n{self.context}\n\nQuestion:\n{self.query}\nAnswer:"
            self.answer = call_llm(prompt, temperature=0.4)
            self.state = extract_features(self.answer, self.context, self.query)

        if action == 2:  # reject
            reward = -5
            done = True
            return self.state, reward, done, False, {}

        doc_sim = self.state[2]
        web_sim = self.state[3]

        reward = 5 * doc_sim + 5 * web_sim - 0.5 * self.step_count

        if action == 0 or self.step_count >= self.max_steps:
            done = True

        return self.state, reward, done, False, {}