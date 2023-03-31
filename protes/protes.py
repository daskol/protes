from functools import partial
from itertools import count
from os import PathLike
from typing import IO, Callable, Optional
from warnings import warn

import jax
import jax.numpy as jnp
from optax import adam, apply_updates

try:
    import cloudpickle
except ImportError:
    cloudpickle = None
    warn(('Package cloudpickle is missing. It is required for loading and '
          'saving sampler object with closures internally.'), ImportWarning)

from protes import tt
from protes.tt import TensorTrain, TensorTrainDensity


@jax.tree_util.register_pytree_node_class
class TensorTrainSampler:
    """Class TensorTrainSampler implements PROTES algorithm as a stateful
    object. It provides two methods :py:`sample` and :py:`submit` which
    essentially corresponds to PROTES two-staged optimization procedure (see
    :py:`minimize`).

    Attributes:
        best tuple[jax.Array, jax.Array]: Best trial. Tuple of minimizer and
            minimum value.
        pdf TensorTrainDensity: Probability density estimator based on
            TT-representation.
    """

    def __init__(self, shape, *, topn=5, rank=5, n_steps=100, opt=None,
                 pdf: Optional[TensorTrainDensity] = None, key=None):
        self.shape = shape
        self.ndim = len(shape)
        self.ranks = (1, ) + (rank, ) * (self.ndim - 1) + (1, )
        self.topn = topn
        self.n_steps = n_steps

        if key is None:
            key = jax.random.PRNGKey(42)

        self.pdf = pdf
        if self.pdf is None:
            train = tt.uniform(key, self.shape, self.ranks)
            self.pdf = TensorTrainDensity.from_train(train)

        self.opt = opt or adam(1e-4)
        self.opt_state = self.opt.init(self.pdf.train)

        self.indicies = jnp.empty((self.topn, self.ndim), jnp.int32)
        self.values = jnp.empty(self.topn)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    @property
    def best(self) -> tuple[jax.Array, jax.Array]:
        """Best trial. A tuple of minimier and minimum."""
        return self.indicies[0], self.values[0]

    @classmethod
    def load(cls, where: IO | PathLike | str):
        """Deserialize sampler from pickled file or file-object.

        Args:
            where: Path to file or file-object.
        """
        if isinstance(where, PathLike | str):
            with open(where, 'rb') as fileobj:
                return cls.load(fileobj)
        return cloudpickle.load(where)

    def save(self, where: IO | PathLike | str):
        """Serialize sampler with pickle to file or file-object.

        Args:
            where: Path to file or file-object.
        """
        if isinstance(where, PathLike | str):
            with open(where, 'wb') as fileobj:
                return self.save(fileobj)
        cloudpickle.dump(self, where)

    def _maximize(self, batch: jax.Array):
        """Maximize log-likelihood on given samples."""

        @partial(jax.value_and_grad, has_aux=True)
        def objective(train, samples):
            pdf = TensorTrainDensity.from_train(train)
            nll = -pdf.score(samples).mean()
            return nll, {}

        def body_fn(i, carry):
            opt_state, train, _ = carry
            (loss, _), loss_grad = objective(train, batch)
            updates, opt_state = self.opt.update(loss_grad, opt_state, train)
            train = apply_updates(train, updates)
            return opt_state, train, {'loss': loss}

        init = (self.opt_state, self.pdf.train, {'loss': float('nan')})
        fini = jax.lax.fori_loop(0, self.n_steps, body_fn, init)

        self.opt_state = fini[0]
        self.pdf = TensorTrainDensity.from_train(fini[1])
        return fini[2]

    def sample(self, key, shape=()) -> jax.Array:
        """Request a single multi-index batch of multi-indices with specified
        shape to explore.

        Args:
            key: PRNG-key.
            shape: Shape of leading dimensions of resulting array of indices.
                By default, a single index is sampled.

        Returns:
            An array of indices with leading dimensions of :py:`shape`.
        """
        return self.pdf.sample(key, shape)

    def submit(self, indices: jax.Array, values: jax.Array):
        """Provides the sampler with the evaluation of a value for a indices.

        Args:
            indicies: Batch of indices where function was evaluated.
            values: Batch of function values corresponding to :py:`indices`.

        Return:
            An updated instance of sampler and auxiliary dictionary which
            contains information about optimization step.
        """
        if indices.shape[:-1] != values.shape:
            raise ValueError('Shape of leading dimensions of indices array '
                             'must match to the shape of values array.')

        # E: Refine empirical distribution of function values.
        ix = values.argsort()[:self.topn]
        self.indicies = indices[ix]
        self.values = values[ix]

        # M: Maximize log-likelihood on sampled indicies.
        aux = self._maximize(self.indicies)
        return self, aux

    def tree_flatten(self):
        leaves = (self.pdf, self.opt_state, self.indicies, self.values)
        treedef = {
            'shape': self.shape,
            'rank': self.ranks[1],
            'topn': self.topn,
            'n_steps': self.n_steps,
            'opt': self.opt,
        }
        return leaves, treedef

    @classmethod
    def tree_unflatten(cls, treedef, leaves):
        sampler = cls(pdf=leaves[0], **treedef)
        sampler.opt_state = leaves[1]
        sampler.indicies = leaves[2]
        sampler.values = leaves[3]
        return sampler


def minimize(fn, shape, max_trials: int, batch_size=50, topn=5, rank=5,
             n_steps=100, opt=None,
             pdf: Optional[TensorTrainDensity] = None,
             train: Optional[TensorTrain] = None, key=None,
             callback_fn: Callable[..., None] = None):
    """Apply PROTES (PRobability Optimizer with TEnsor Sampling) algorihtm for
    minimization of vectorized function :py:`fn`.

    >>> (x, y), pdf = minimize(
    >>>    fn=lambda ix: ix.sum(axis=1),
    >>>    shape=(5, ) * 10,
    >>>    max_trials=5000
    >>> )
    >>> print(x)
    [0 0 0 0 0 0 0 0 0 0]
    >>> print(y)
    0

    Args:
        fn: Objective function.
        shape: Shape of discrete search space. It bounds indices passed to
            objective function.
        max_trials: Maximal number of evaluations of objective function.
        batch_size: Number of different indices passed simulteneously to
            objective function for evaluation.
        topn: Maximal number of best trials to store for likelihood estimation.
        rank: Rank of TT-tensor of density estimator. It should be a number
            since TT-ranks is assumed to be uniform.
        opt: Optax-like optimizer. If it is not specified then used Adam with
            learning rate 1e-4.
        pdf: Initial PDF-estimator in TT-format. If it is not specified then
            random estimator is generated with :py:`key`.
        key: PRNG-key used for sampling and building initial density estimator.
        callback_fn: Callback function is invoked at the end of each iteration.

    Returns:
        Resulting 2-tuple consists of pair of minimizer and minimum and density
        estimator.
    """
    if max_trials <= 0:
        raise ValueError('Maximal number of trials must be positive.')

    if pdf is not None and train is not None:
        warn(('Initial tensor train value is ignored since initial density '
              'estimator is specified.'), RuntimeWarning)

    if key is None:
        key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    sampler = TensorTrainSampler(shape, topn=topn, rank=rank, n_steps=n_steps,
                                 opt=opt, pdf=pdf, key=subkey)

    # TODO(@daskol): Direct jitting of bounded method causes recompilation. So,
    # we need to wrap bounded method with free functions. See
    # https://github.com/google/jax/issues/15338 for details.

    @partial(jax.jit, static_argnums=2)
    def sample_fn(key, sampler, batch_size):
        return sampler.sample(key, (batch_size, ))

    @jax.jit
    def submit_fn(sampler, xs, ys):
        return sampler.submit(xs, ys)

    num_trials = 0
    for it in count():
        # Check how many sampled indices we can calculate.
        if (remainder := max_trials - num_trials) <= 0:
            break
        batch_size = min(batch_size, remainder)
        num_trials += batch_size

        # E: Sample a batch of indicies/values with density estimator.
        key, subkey = jax.random.split(key)
        xs = sample_fn(key, sampler, batch_size)
        ys = fn(xs)

        # M: Do several gradient optimization steps in order to maximize
        # likelihood of samples which minimize objective.
        sampler, aux = submit_fn(sampler, xs, ys)

        if callable(callback_fn):
            callback_fn(it, sampler.pdf, sampler.indicies, sampler.values, aux)

    return sampler.best, sampler.pdf
