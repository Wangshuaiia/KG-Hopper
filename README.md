# KG-Hopper: Empowering Compact Open LLMs with Knowledge Graph Reasoning via Reinforcement Learning

KG-Hopper is a novel Reinforcement Learning (RL) framework that empowers compact open LLMs with the ability to perform integrated multi-hop reasoning within a single inference round. Rather than reasoning step-by-step, we train a Reasoning LLM that embeds the entire KG traversal and decision process into a unified “thinking” stage, enabling global reasoning over cross-step dependencies and dynamic path exploration with backtracking. KG-Hopper, based on a 7B-parameter LLM, consistently outperforms larger multi-step systems (up to 70B) and achieves competitive performance with proprietary models such as GPT-3.5-Turbo and GPT-4o-mini.

### Package Directory Structure
```
.
├── OpenRLHF-RAG
├── README.md
├── data
├── evaluation
│   ├── eval_search_kg.py
│   └── extract_entity_from_query.py
├── kg-tool
├── requirements.txt
├── reward-remote
│   └── reward_server_qwen_zero.py
└── scripts
    ├── ray_start.sh
    └── reinforce_train.sh
```
---

## Project Structure

- **`OpenRLHF-RAG/`**  
  Contains tools for RLHF (Reinforcement Learning with Human Feedback) model training.  
  This folder is adapted from [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF).

- **`data/`**  
  Stores the training and testing datasets.

- **`evaluation/`**  
  Used to load trained models and perform evaluation.

- **`kg-tool/`**  
  Provides utilities for retrieving information from a knowledge graph.

- **`reward-remote/`**  
  Implements the remote reward function used during RL training.

- **`scripts/`**  
  Contains training scripts used in the RL training pipeline.

---

## Requirements
- Python 3.x
- Install the required libraries:
  ```bash
  pip install -r requirements.txt
## Usage

1. Enter the **KG-Hopper** folder:
   ```bash
   cd KG-Hopper
   ```
2. Training：
   ```bash
    ## Ray start
    bash scripts/ray_start.sh

    ## Start Reward Server
    python reward-remote/reward_server.py --port 1278

    ## Training
    bash scripts/reinforce_train.sh
    ```
3. Evaluation：
   ```bash
    python evaluation/eval_search_kg.py
   ```

