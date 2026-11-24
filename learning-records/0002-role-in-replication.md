# Role: implements SPP end-to-end on Levanter (data pipeline + training); collaborator owns evals

The user is porting SPP from the Megatron reference (`spp-training`) to Marin/Levanter: reflection insertion, attention masking, RoPE aliasing, and the data pipeline. Their TPU-scaling role is not yet defined. Priority topics: tokenization (in depth), RoPE, attention masking, data mixing. They want the first three covered within a few days of 2026-11-23.

**Implications:** Lessons should end in Levanter/JAX code the user will actually write. Prefer JAX exercises over PyTorch. Evals content is low priority.
