# Mission: LLM Pretraining

## Why
I work in AI safety and am moving into character training, i.e. building alignment in from token zero of pretraining (e.g. Synthetic Persona Pretraining). **Near-term:** I'm contributing to [Marin](https://marin.community/). We're replicating the SPP paper on Marin's stack (JAX/Levanter on TPUs), then scaling to a **1.4B-parameter run at 20 tokens per parameter**, and maybe further. **Long-term:** I want to pretrain models from scratch with real understanding, and do original research on pretraining-time alignment.

## Success looks like
- **(Near-term)** Reason confidently about the Marin SPP replication and the 1.4B / 20 TPP run: compute budget, data mixture, hyperparameters, and what the loss curves and evals mean. Contribute code and judgment instead of just following along.
- Write a small GPT from scratch and understand every line, then train it locally (Mac/MLX) or on free Colab/Kaggle GPUs.
- Implement a modern frontier-style architecture from scratch (MoE, MLA/linear attention, e.g. the Kimi K2/K3 family) and train a scaled-down version on rented GPUs as a capstone.
- Read pretraining papers fluently (scaling laws, data mixtures, technical reports), check their math, and come up with my own ideas.
- Have enough mathematical depth to propose and test new pretraining-time alignment interventions.

## Constraints
- 5–10 hrs/week, sometimes more before a deadline. Target: very good at pretraining in about 3 months (by late Feb 2027).
- Compute: Mac (Apple Silicon) first, also interested in systems/kernel work like MLX. Free Colab/Kaggle T4s are fine. Rented GPUs (Lambda/Vast/Modal) only for capstone work.
- Background is broad but sparse: comfortable in Python, has done classical ML/DL (Andrew Ng), autograd from scratch, and Attention Is All You Need. Recognises most LLM terms but can't yet implement or reason about them in depth.

## Out of scope (for now)
- Post-training beyond what SPP needs (RLHF/DPO depth), inference serving, quantization for deployment.
