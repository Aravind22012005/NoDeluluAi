Flowchart-
                    ┌────────────────────┐
                    │     User Query     │
                    └──────────┬─────────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │    RAG Retrieval   │
                    │   (document.txt)   │
                    └──────────┬─────────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │    LLM Generator   │
                    │    (OpenAI API)    │
                    └──────────┬─────────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │  Feature Extractor │
                    │  • Length Check    │
                    │  • Doc Similarity  │
                    │  • Web Similarity  │
                    │  • Contradiction   │
                    └──────────┬─────────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │    RL Controller   │
                    │     (PPO Policy)   │
                    └──────────┬─────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
   Accept Answer         Regenerate           Reject / Retrieve
                               │
                               ▼
                    ┌────────────────────┐
                    │     Final Answer   │
                    └────────────────────┘
                    
 Steps to Run the Project-
1)Install dependencies
   pip install -r requirements.txt
2)Set your OpenAI API key
   Open config.py
   Add your API key in the appropriate field
3)Train the RL Policy
   python train.py
   This generates:
        rl_guard_policy.zip (Contains the trained PPO policy parameters)
4)Run the Application
   python app.py
   Open the URL generated in the terminal
5)Start testing queries
 
