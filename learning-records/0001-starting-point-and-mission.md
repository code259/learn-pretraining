# Starting point: broad-but-sparse background, mission anchored on Marin SPP replication

The user does AI safety research and is moving into character training / pretraining-time alignment. Their near-term work is replicating Synthetic Persona Pretraining on Marin (JAX/Levanter, TPUs), then a 1.4B run at 20 TPP. They report: comfortable in Python; Andrew Ng-level classical ML/DL; has written autograd from scratch; has worked through *Attention Is All You Need*; recognises nearly all LLM vocabulary but can't implement or reason about it in depth.

**Implications:** Don't re-teach backprop basics or what attention is. Do verify depth with retrieval questions, because recognition ≠ fluency here. Ground early lessons in the concrete 1.4B / 20 TPP run, and use TPU numbers (scaling book) rather than GPU numbers by default.
