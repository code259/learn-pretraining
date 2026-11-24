# LLM Pretraining Resources

## Knowledge

### Mission-specific
- [Paper: _Synthetic Persona Pretraining: Alignment from Token Zero_, Minder, Singhal et al. (EPFL DLAB), arXiv:2608.13482](https://arxiv.org/abs/2608.13482)
  The paper we're replicating on Marin. Annotates pretraining docs with constitution-grounded first-person reflections, then uses post-training "persona binding". Experiments go up to 3B params / 500B tokens. Use for: everything SPP.
- [Code: epfl-dlab/spp](https://github.com/epfl-dlab/spp): umbrella repo (spp-data, spp-training, spp-evals). Includes reflection insertion, attention masking, and RoPE-aliasing variants. Use for: replication details.
  - Key file: [`spp-training/pretraining/dataloaders/spp_dataloader.py`](https://github.com/epfl-dlab/spp-training/blob/main/pretraining/dataloaders/spp_dataloader.py) (`_make_spp_sample`) is the reference implementation of insertion, the attention mask, RoPE aliasing, and the loss masks. [`docs/04-pretraining.md`](https://github.com/epfl-dlab/spp-training/blob/main/docs/04-pretraining.md) is the spec.
- Levanter files to port into: `lib/levanter/src/levanter/models/lm_model.py` (`LmExample`), `layers/attention_mask.py` (`AttentionMask`), `layers/rotary.py`, `models/llama.py`.
- [LessWrong post on SPP](https://www.lesswrong.com/posts/3xQQK9i8mhJDE2uMg/synthetic-persona-pretraining-alignment-from-token-zero): author summary plus discussion.
- [Paper: _Alignment Pretraining: AI Discourse Causes Self-Fulfilling (Mis)alignment_, arXiv:2601.10160](https://arxiv.org/abs/2601.10160): related pretraining-time alignment work.
- [Marin](https://marin.community/) and the [Marin monorepo](https://github.com/marin-community/marin). Levanter lives at `lib/levanter/` (merged Nov 2025). [Levanter docs](https://levanter.readthedocs.io). Use for: the actual stack (JAX, named tensors, Ray, TPU slices, deterministic data).
- [Google blog: Marin, first fully open model developed using JAX](https://developers.googleblog.com/en/stanfords-marin-foundation-model-first-fully-open-model-developed-using-jax/): overview of Marin-8B and Levanter's design.

### Core pretraining knowledge
- [Book: _How to Scale Your Model_, Austin et al. (Google DeepMind)](https://jax-ml.github.io/scaling-book). **The** reference for TPU + JAX + transformer math: FLOPs, rooflines, sharding, worked LLaMA-3 example. Directly matches Marin's hardware. Use for: compute budgets, MFU, parallelism, TPU specs.
- [Course: Stanford CS336 _Language Modeling from Scratch_ (Hashimoto & Liang)](https://cs336.stanford.edu/): lectures on YouTube and [GitHub](https://github.com/stanford-cs336). Assignments build tokenizer → transformer → systems → scaling → data. From the Marin team's own group (Percy Liang). Use for: the backbone curriculum. [Lec 1: Overview and Tokenization (video)](https://www.youtube.com/watch?v=SQ3fZ1sAqXI). Lec 3 (Architectures, Hyperparameters) is linked from the course page and covers pre-norm, RMSNorm, SwiGLU and RoPE.
- [Video: Karpathy, _Let's build the GPT Tokenizer_](https://www.youtube.com/watch?v=zduSFxRajkE) and [minbpe](https://github.com/karpathy/minbpe): the best deep dive on BPE, GPT-2/4 regexes, and tokenizer pathologies.
- [Paper: Sennrich et al. 2016, BPE for NMT](https://arxiv.org/abs/1508.07909) · [GPT-2 paper §2.2 (byte-level BPE)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) · [gpt-2 encoder.py](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
- [SmolLM2 tokenizer.json](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/blob/main/tokenizer.json): ground truth. BPE, 49,152 vocab (17 specials, 235 byte tokens, 48,900 merges); pre-tokenizer = Digits(individual) → ByteLevel(add_prefix_space=false).
- [Paper: _Training Compute-Optimal LLMs_ (Chinchilla), Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556). Chinchilla is 70B params / 1.4T tokens, the origin of "20 tokens per parameter". Use for: scaling laws.
- [Guide: HF _Smol Training Playbook_](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook): end-to-end SmolLM3 (3B, 11T tokens) story covering ablations, data mixtures, architecture choices, and loss-spike debugging. Use for: practitioner judgment.
- [Code: karpathy/nanochat](https://github.com/karpathy/nanochat) and [nanoGPT](https://github.com/karpathy/nanoGPT), plus the "Zero to Hero" video "Let's reproduce GPT-2 (124M)". Use for: from-scratch implementation milestones.

### Modern architectures (long-term)
- [Kimi K3 repo (Moonshot AI)](https://github.com/MoonshotAI/Kimi-K3): 2.8T-param MoE (16 of 896 experts active), Kimi Delta Attention (hybrid linear attention), Attention Residuals. Use for: capstone architecture target. Verify details against the tech report, since secondary sources disagree.

## Wisdom (Communities)
- Marin's own channels (the user is already contributing). Use for: stack questions and run reviews. *TODO: record the link to the community chat once the user confirms which one.*
- [EleutherAI Discord](https://discord.gg/eleutherai): long-running open pretraining research community. Use for: scaling/interp/data discussion.
- [GPU MODE Discord](https://discord.gg/gpumode): kernels and systems. Use for: MLX/Pallas/CUDA kernel questions.

## Gaps
- A trusted MLX-specific resource for training-loop performance on Apple Silicon (peak FLOPs per chip, kernel authoring).
- Primary source for Marin's 1.4B run config (hardware type, slice size, tokenizer, data mix). Ask the user or check the Marin repo/issues.
