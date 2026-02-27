Flow chart-
                ┌──────────────────┐
                │    User Query    │
                └─────────┬────────┘
                          │
                          ▼
                ┌──────────────────┐
                │   RAG Retrieval  │
                │  (document.txt)  │
                └─────────┬────────┘
                          │
                          ▼
                ┌──────────────────┐
                │   LLM Generator  │
                │   (OpenAI API)   │
                └─────────┬────────┘
                          │
                          ▼
                ┌──────────────────┐
                │ Feature Extractor│
                │  - Length        │
                │  - Doc Similarity│
                │  - Web Similarity│
                │  - Contradiction │
                
                └─────────┬────────┘
                          │
                          ▼
                ┌──────────────────┐
                │   RL Controller  │
                │  (PPO Policy)    │
                └─────────┬────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
    Accept Answer    Regenerate       Reject / Retrieve
                          │
                          ▼
                ┌──────────────────┐
                │   Final Answer   │
                └──────────────────┘
Steps to run the code after cloning the repo-
1)Install requirements.txt to install all the pre-requisites
2) In config.py set your openai api key
3) run train.py which would create rl_guard_policy.zip file. This consists of all the training parameters.
4) run run.py and follow the url that would be generated in the terminal.
