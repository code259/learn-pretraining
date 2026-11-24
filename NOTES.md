# Teaching Notes

## Learner preferences
- Wants mathematical depth and the ability to *reason* and generate ideas, not just recognise terms.
- Likes systems/perf work (MLX kernels, squeezing the Mac) as a source of hands-on exercises.
- Background is "broad but sparse". Don't assume depth just because they recognise a term. Probe with retrieval questions before skipping anything.
- Time varies week to week. Occasionally a deadline means they need a targeted crash lesson on one thing.

- **Propose the lesson plan and get a go-ahead before building lessons.** Don't pick topics on my own when the user may have specific needs. (2026-11-23: I built a compute-budget lesson before asking, and the user had other priorities.)

- **Be explicit about expectations.** Say what's a map vs a deep dive, and mark each reading "do now" vs "optional". Don't word "ask me" boxes as if the user should already understand things that haven't been taught. (2026-11-23: L02's wording confused the user about whether they were supposed to understand pre-norm/GQA/rotate_half already, and whether CS336 was required.)

## Deferred (after the four SPP topics)
Architecture deep dive: pre-norm vs post-norm, RMSNorm, GQA/MQA, SwiGLU, tied embeddings. Then optimizers (AdamW), LR schedules (warmup, WSD), stability, and scaling laws.

## Current priorities (user-stated 2026-11-23, for the SPP replication)
Tokenization (in depth), RoPE, attention masking, data mixing. Start with a brief architecture overview. Spread each across multiple lessons for depth. Long-term the user wants *everything* (optimizers, warmup/schedules, the literature), but these four come first.

## Confirmed plan (2026-11-23)
Pace: RoPE, masking and tokenization within the next few days. No hard deadline. The user implements SPP in Levanter plus the data pipeline. A collaborator does evals. Exercises use **JAX** and point at real Levanter/SPP code.
- L01 (side): 6ND compute budget ✅ built
- L02: Architecture map ✅ built
- L03: Tokenization I: byte-level BPE from scratch ✅ built (exercise in exercises/03-bpe; reference solution verified to pass, kept in scratchpad only)
- L04: Tokenization II: special tokens, packing vs padding, loss masks; SmolLM2 tokenizer + `<assistant>` (spp-training `pretraining/extend_tokenizer.py`); Megatron tokens/labels vs Levanter `loss_weight[i]` = predict i+1
- L05: RoPE I: derivation, implement in JAX, verify relative property. Levanter `rotary.py` uses the rotate_half (NeoX/HF) convention, inv_freq = 1/theta^(2i/d)
- L06: RoPE II: position IDs as input; SPP aliasing; Levanter `LmExample` has NO pos_ids field, but the model `__call__` takes `pos_ids`, so they need threading through
- L07: Masking I: causal + segment IDs (Levanter `AttentionMask.with_segment_ids`, splash attention on TPU)
- L08: Masking II: SPP block mask. It isn't a segment-ID mask; options: explicit_mask (materialized S×S, slow) vs a smarter structured mask. Revisit the "r_N → first post-ctx token" question from L02. Also `reflection_precontext_mask_pct` (random pre-context drop for reflection rows)
- L09–10: Data mixing: compact 80.6% / annotated 19.4% / canary; packed vs padded; midtraining step-count subtlety; DoReMi/RegMix literature; Marin mixture config

## SPP implementation facts (from reading the code, 2026-11-23)
- `spp-training/pretraining/dataloaders/spp_dataloader.py::_make_spp_sample` is the single source of truth for tokens/labels/loss_mask/reflection_mask/attention_mask(+bias)/position_ids.
- Positions: pre `0..n-1`, reflection block `n..n+I-1`, post `n..` (aliased). Reflection rows attend causally to the prefix.
- Loss: `loss_mask[n-1]=0` (never predict `<assistant>`); `<assistant>→r1` IS trained; `rN → first post token` is in reflection_mask; padding (token 0) masked.
- Paper: warmup 3,000 vs launchers 2,000 (unresolved discrepancy noted in the repo docs).

## Spacing queue
- Start L04 with recall from L03: encode rule (lowest rank first), leading-space rule, why splice IDs rather than retokenize. Collect the user's answer on the mid-word insertion point question.
- L04 gotcha to teach: HF tokenizers match added special-token strings in raw text (a web doc containing literal "<|endoftext|>" or "<assistant>"). Check Levanter's tokenization path for split_special_tokens behaviour.
- Start L03 with cold recall from L02: write SPP position IDs for a new example; attention vs loss mask distinction.
