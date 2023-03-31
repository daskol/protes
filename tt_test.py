from functools import partial

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from tt import TensorTrain, TensorTrainDensity, categorical


class TestTensorTrainDensity:

    def test_sample():
        key = jax.random.PRNGKey(42)
        cores = [jax.random.uniform(key, (1, 3, 1)) for _ in range(3)]
        train = TensorTrain.from_cores(cores)
        pdf = TensorTrainDensity.from_train(train)
        assert len(pdf.interfaces) == train.ndim

        @partial(jax.jit, static_argnums=1)
        def sample(key, shape, pdf):
            samples = pdf.sample(key, shape)
            return samples

        shape = (1, 2, 3)
        samples = sample(key, shape, pdf)
        assert samples.shape[:-1] == shape
        assert samples.shape[-1] == train.ndim


    def test_score():
        # Generate uniform distribution of indices.
        pdf = TensorTrainDensity.uniform((3, 3, 3), (1, 1, 1, 1))
        assert len(pdf.interfaces) == pdf.ndim

        # Generate all possible indices.
        indices = jnp.mgrid[:3, :3, :3].reshape(3, -1).T
        log_probas = pdf.score(indices)
        assert log_probas.ndim == 1
        assert log_probas.shape == indices.shape[:1]

        probas = jnp.exp(log_probas)
        assert_allclose(probas, 1 / 3**3, atol=1e-6)
