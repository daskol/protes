from functools import partial
from itertools import count
from typing import Callable, Optional
from warnings import warn

import jax
from optax import adam, apply_updates

import tt
from tt import TensorTrain, TensorTrainDensity


class TensorTrainSampler:

    def __init__(self, shape, *, topn=5, n_steps=100, rank=5):
        self.shape = shape
        self.topn = topn
        self.n_steps = n_steps
        self.rank = rank

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def sample(self, key, shape=()) -> jax.Array:
        raise NotImplementedError

    def submit(self, samples, values):
        if samples.shape[:-1] != values.shape:
            raise ValueError('Shape of leading dimensions of samples array '
                             'must match to the shape of values array.')
        return self


def minimize2(fn, shape, max_trials, batch_size, topn, n_steps, rank,
              key):
    sampler = TensorTrainSampler(shape, topn=topn, n_steps=n_steps, rank=rank)
    num_trials = 0
    for _ in count():
        if (remainder := max_trials - num_trials) > 0:
            batch_size = max(batch_size, remainder)
            num_trials += batch_size
        else:
            break

        key, subkey = jax.random.split(key)
        indices = sampler.sample(key, (batch_size, ))
        values = fn(indices)
        sampler = sampler.submit(indices, values)
    return values, sampler


def minimize(fn, shape, max_trials: int, batch_size=50, topn=5, rank=5,
             n_steps=100, opt=None,
             density: Optional[TensorTrainDensity] = None,
             train: Optional[TensorTrain] = None, key=None,
             callback_fn: Callable[..., None] = None):
    """Apply PROTES (PRobability Optimizer with TEnsor Sampling) algorihtm for
    minimization of vectorized function :py:`fn`.

    :return: Resulting 2-tuple consists of pair of minimizer and minimum and
    density estimator.
    """
    ndim = len(shape)
    ranks = (1, ) + (rank, ) * (ndim - 1) + (1, )

    if max_trials <= 0:
        raise ValueError('Maximal number of trials must be positive.')

    if key is None:
        key = jax.random.PRNGKey(42)

    if density is None:
        if train is None:
            key, subkey = jax.random.split(key)
            train = tt.uniform(subkey, shape, ranks)
        density = TensorTrainDensity.from_train(train)
    elif train is not None:
        warn(('Initial tensor train value is ignored since initial density '
              'is specified.'), RuntimeWarning)

    if opt is None:
        opt = adam(1e-4)
    opt_state = opt.init(density.train)

    @partial(jax.value_and_grad, has_aux=True)
    def objective(train, samples):
        pdf = TensorTrainDensity.from_train(train)
        nll = -pdf.score(samples).mean()
        return nll, {}

    @partial(jax.jit, static_argnums=2)
    def expect(key, pdf: TensorTrainDensity, size: int):
        # Sample a batch of indicies from estimated distribution and evaluate
        # function.
        xs = pdf.sample(key, (size, ))
        ys = fn(xs)
        # Keep only samples with the lowest value.
        ix = ys.argsort()[:topn]
        return xs[ix], ys[ix]

    @jax.jit
    def maximize(opt_state, pdf: TensorTrainDensity, batch: jax.Array):
        def body_fn(i, carry):
            opt_state, train, _ = carry
            (loss, _), loss_grad = objective(train, batch)
            updates, opt_state = opt.update(loss_grad, opt_state, pdf.train)
            train = apply_updates(train, updates)
            return opt_state, train, {'loss': loss}
        init = (opt_state, pdf.train, {'loss': float('nan')})
        opt_state, train, *tail = jax.lax.fori_loop(0, n_steps, body_fn, init)
        pdf = TensorTrainDensity.from_train(train)
        return opt_state, pdf, *tail

    for it in range(max_trials // batch_size):
        key, subkey = jax.random.split(key)

        # E: Sample a batch of indicies/values with density estimator.
        indicies, values = expect(subkey, density, batch_size)

        # M: Do several gradient optimization steps in order to maximize
        # likelihood ofsamples which minimize objective.
        opt_state, density, aux = maximize(opt_state, density, indicies)

        if callable(callback_fn):
            callback_fn(it, density, indicies, values, aux)

    best = (None, None)
    if values.size:
        best = (indicies[0, ...], values[0])
    return best, density
