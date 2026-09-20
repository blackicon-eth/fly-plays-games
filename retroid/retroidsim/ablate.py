"""Ways to damage the connectome, for the ablation controls.

- `none`: the real, unmodified weights.
- `shuffle`: permute the *weights* among the existing connections. The wiring
  (which neuron talks to which) is unchanged; only the calibrated strengths are
  scrambled. This is a weak ablation: the network stays alive and can even respond
  more strongly.
- `rewire`: replace the *topology*. Every neuron keeps the same number of incoming
  connections, but they now come from random neurons. This tests whether the
  specific wiring matters, not just the weights.
- `silence`: zero all weights. A strong ablation: the neurons can no longer talk.

Each one is followed by retraining the readout, so the scrambled brain gets the
same benefit of a trained decoder as the real one.
"""
from __future__ import annotations

import numpy as np


def ablate(brain, mode: str, seed: int = 0) -> None:
    if mode == "none":
        return
    if mode == "shuffle":
        np.random.default_rng(seed).shuffle(brain.weights)
        return
    if mode == "silence":
        brain.weights[:] = 0.0
        return
    if mode == "rewire":
        rng = np.random.default_rng(seed)
        indptr, n = brain.indptr, brain.n
        indices = np.empty_like(brain.indices)
        for c in range(n):
            k = int(indptr[c + 1] - indptr[c])
            if k:
                indices[indptr[c]:indptr[c + 1]] = rng.integers(0, n, k, dtype=brain.indices.dtype)
        brain.indices = indices
        return
    raise ValueError(f"unknown ablation: {mode}")
