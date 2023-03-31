"""Package protes implements PROTES algorithm in Python with JAX.

[1]: Batsheva et al -- PROTES: Probabilistic Optimization with Tensor Sampling
     // 2023. https://arxiv.org/abs/2301.12162
"""

from protes.protes import TensorTrainSampler, minimize  # noqa: F401
from protes.tt import TensorTrain, TensorTrainDensity  # noqa: F401
