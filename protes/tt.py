from functools import partial, reduce
from typing import Any, List, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class TensorTrain:
    """Class TensorTrain is a core data structure in implementation of
    TensorTrain in JAX. It is a base class for any specific specific algebraic
    type (like TT-matrix or TT-tensor) which leverage TensorTrain
    representation underneath.

    :param cores: List of TT-cores.
    :param shape: Tensor shape.
    :param ranks: TT-rank.

    We maintain the following invariants in this class.

     1. All dtypes of cores are the same.
     2. Number of dimensions, number of cores, number of elements in shape are
        the same. Number of elements in ranks is greater by one.
     3. There is a degenerate case of trivial (empty) shape which corresponds
        to scalar.
    """

    def __init__(self, cores: List[jnp.array], shape, ranks, dtype=None):
        if len(cores) != len(shape) and len(cores) != len(ranks) - 1:
            raise ValueError('Number of dimensions is inconsistent.')

        self.shape = tuple(shape)
        self.ranks = tuple(ranks)
        self.cores = cores
        self.dtype = dtype or self.cores[0].dtype

        # TODO: Remove in the future. Rank should be effective rank.
        self.rank = self.ranks

    def __repr__(self) -> str:
        params = ', '.join([
            f'ndim={self.ndim}',
            f'shape={self.shape}',
            f'ranks={self.ranks}',
            f'dtype={self.dtype}',
        ])
        return f'{self.__class__.__name__}({params})'

    def __add__(self, other) -> 'TensorTrain':
        return add(self, other)

    def __eq__(lhs, rhs) -> bool:
        if lhs.shape != rhs.shape or lhs.rank != rhs.rank:
            return False
        for lc, rc in zip(lhs.cores, rhs.cores):
            if jnp.all(lc != rc):
                return False
        return True

    def __getitem__(self, ix):
        acc = self.cores[0][0, ix[0]]
        for i, core in zip(ix[1:], self.cores[1:]):
            acc = acc @ core[:, i, :]
        return acc.squeeze()

    @property
    def ndim(self):
        return len(self.cores)

    @property
    def size(self):
        return reduce(lambda x, y: x + np.prod(y.shape), self.cores, 0)

    @classmethod
    def from_cores(cls, cores: List[jnp.array]) -> 'TensorTrain':
        dtype = cores[0].dtype
        for i, core in enumerate(cores[1:]):
            if dtype != core.dtype:
                raise ValueError('Cores\' dtype is inconsistent: '
                                 f'dtype of core #0 ({dtype}) does not '
                                 f'match dtype of core #{i}.')
        shape = [core.shape[1] for core in cores]
        ranks = [core.shape[0] for core in cores]
        ranks.append(cores[-1].shape[2])
        return TensorTrain(cores, shape, ranks, dtype)

    def tolist(self):  # TODO: Choose proper name.
        return [core.tolist() for core in self.cores]

    def orthogonalize(self) -> 'TensorTrain':
        raise NotImplementedError

    def truncate(self, eps=1e-5, rmax=1000000) -> 'TensorTrain':
        raise NotImplementedError

    def tree_flatten(self):
        return self.cores, {
            'shape': self.shape,
            'ranks': self.ranks,
            'dtype': self.dtype,
        }

    @classmethod
    def tree_unflatten(cls, treedef, leaves):
        return cls(leaves, **treedef)


def add(lhs: TensorTrain, rhs):
    if isinstance(rhs, (np.bool_, np.number, bool, int, float, complex)):
        raise NotImplementedError
    elif isinstance(rhs, TensorTrain):
        if lhs.shape != rhs.shape:
            raise ValueError('Shape of operands mismatch: '
                             f'{lhs.shape} != {rhs.shape}.')
        first = jnp.concatenate([lhs.cores[0], rhs.cores[0]], axis=2)
        cores = [first]
        # TODO: Use jax.lax.scan for iterations.
        for lhs_core, rhs_core in zip(lhs.cores[1:-1], rhs.cores[1:-1]):
            size, *lhs_ranks = lhs_core.shape
            size, *rhs_ranks = rhs_core.shape
            top_core = jnp.zeros((size, lhs_ranks[0], rhs_ranks[1]))
            bot_core = jnp.zeros((size, rhs_ranks[0], lhs_ranks[1]))
            core = jnp.block([[lhs_core, top_core],
                              [bot_core, rhs_core]])
            cores.append(core)
        last = jnp.concatenate([lhs.cores[-1], rhs.cores[-1]], axis=1)
        cores.append(last)
        return TensorTrain.from_cores(cores)
    else:
        raise ValueError(f'Wrong type of the right operand: {type(rhs)}.')


def _make_tensor_train(shape, ranks, dtype, factory_fn):
    cores = []
    for core_shape in zip(ranks[:-1], shape, ranks[1:]):
        core = factory_fn(core_shape, dtype=dtype)
        cores.append(core)
    return TensorTrain.from_cores(cores)


def empty(shape, ranks, dtype=None) -> TensorTrain:
    return _make_tensor_train(shape, ranks, dtype, jnp.empty)


def full(shape, ranks, fill_value, dtype=None) -> TensorTrain:
    factory_fn = partial(jnp.full, fill_value=fill_value)
    return _make_tensor_train(shape, ranks, dtype, factory_fn)


def ones(shape, ranks, dtype=None) -> TensorTrain:
    return _make_tensor_train(shape, ranks, dtype, jnp.ones)


def zeros(shape, ranks, dtype=None) -> TensorTrain:
    return _make_tensor_train(shape, ranks, dtype, jnp.zeros)


def uniform(key, shape, ranks, minval=0.0, maxval=1.0):
    cores = []
    keys = jax.random.split(key, len(shape))
    for key, *core_shape in zip(keys, ranks[:-1], shape, ranks[1:]):
        core = jax.random.uniform(key, core_shape, minval=minval,
                                  maxval=maxval)
        cores.append(core)
    return TensorTrain.from_cores(cores)


def categorical(key, shape, probas, replace=False):
    """Sample from categorical distribution with unnormalized probabilities
    `probas` with replacement or without it.
    """
    if not isinstance(shape, Sequence):
        raise TypeError(f'Shape must be a sequence, {type(shape)}.')
    probas = jnp.asarray(probas)
    if probas.ndim != 1:
        raise ValueError('Categorical probability must be 1-array, not '
                         f'{probas.ndim}-array.')
    choices = jnp.arange(probas.size)
    size = reduce(lambda x, y: x * y, shape, 1)
    samples = jax.random.choice(key, choices, (size, ), replace, probas)
    return samples.reshape(shape)


@register_pytree_node_class
class TensorTrainDensity:
    """Class TensorTrainDensity implements an empirical density estimation with
    tensor train and fast sampling from corresponding probability density.

    [1]: Dolgov et al -- Approximation and sampling of multivariate probability
         distributions in the tensor train decomposition // 2018.
         https://arxiv.org/abs/1810.01212
    """

    def __init__(self, train: TensorTrain, interfaces: List[jax.Array]):
        self.train = train
        self.interfaces = interfaces

    def __repr__(self) -> str:
        status = 'none' if self.interfaces is None else 'computed'
        params = ', '.join([
            f'ndim={self.train.ndim}',
            f'shape={self.train.shape}',
            f'ranks={self.train.ranks}',
            f'interfaces={status}',
        ])
        return f'{self.__class__.__name__}({params})'

    @property
    def ndim(self):
        return self.train.ndim

    @classmethod
    def from_train(cls, train: TensorTrain):
        # Prepare interface matrices for fast sampling.
        interfaces = [jnp.ones(1)]
        for core in train.cores[:0:-1]:
            margin = core.sum(axis=1)
            interfaces.append(margin @ interfaces[-1])
        interfaces = interfaces[::-1]
        return cls(train, interfaces)

    def sample(self, key: jax.Array, shape: Any) -> jax.Array:
        if not isinstance(shape, Sequence):
            raise ValueError(f'Shape must be a sequence, not {type(shape)}.')

        @partial(jax.vmap, in_axes=(0, 0))
        def sample_fn(key, probas):
            return categorical(key, (), probas, True)

        samples = []
        nosamples = reduce(lambda x, y: x * y, shape, 1)
        keys = jax.random.split(key, self.ndim)
        prefix = jnp.ones((nosamples, 1))
        for key, core, suffix in zip(keys, self.train.cores, self.interfaces):
            # Estimate conditional probabilities.
            pdf = jnp.abs(prefix @ (core @ suffix))
            probas = pdf / pdf.sum(axis=1, keepdims=True)

            # Sample with conditioned unnormalized probabilities.
            subkeys = jax.random.split(key, probas.shape[0])
            ix = sample_fn(subkeys, probas)
            samples.append(ix[:, None])  # Column of output batch.

            # Update conditional probabilities on the left.
            prefix = jnp.einsum('ij,jik->ik', prefix, core[:, ix, :])
        return jnp.hstack(samples).reshape(*shape, -1)

    def score(self, samples: jax.Array) -> jax.Array:
        """Estimate log-likelihood of the given samples.

        :param samples: Batch of indicies in which log-likelihood is desired.
        The last dimension is (coordinate) index dimension.

        :return: An array of log-likelihood calculated at specified indices.
        Shape of result corresponds to the leading dimensions of input array.

        """
        # We assume that indicies are in the last dimension but we need to
        # flatten the leading dimension and iterate over the last one.
        shape = samples.shape[:-1]

        if samples.ndim == 1:
            samples = samples[None, :]
        if samples.shape[-1] != self.ndim:
            raise ValueError('Number of samples dimensions does not match '
                             f'arity of density estimator: ndim={self.ndim}.')

        size = reduce(lambda x, y: x * y, shape, 1)
        samples = samples.reshape(size, self.ndim)

        log_probas = []
        prefix = jnp.ones((size, 1))
        for ix, core, suffix in zip(samples.T, self.train.cores,
                                    self.interfaces):
            # Estimate conditional probabilities.
            pdf = jnp.abs(prefix @ (core @ suffix))
            pdf /= pdf.sum(axis=1, keepdims=True)
            assert pdf.ndim == 2, 'The first dimension is batch dimension.'

            # Instead of sampling here from categorical distribution we take
            # probability estimates for the specified indices.
            log_probas.append(pdf[jnp.arange(ix.size), ix])

            # Update conditional probabilities on the left.
            prefix = jnp.einsum('ij,jik->ik', prefix, core[:, ix, :])
            # NOTE There is such normalization in reference implementation.
            # prefix /= jnp.linalg.norm(prefix, axis=-1, keepdims=True)

        return jnp.hstack(log_probas) \
            .reshape(*shape, self.ndim) \
            .sum(axis=-1)

    def tree_flatten(self):
        return (self.train, self.interfaces), {}

    @classmethod
    def tree_unflatten(cls, treedef, leaves):
        return cls(*leaves, **treedef)

    @classmethod
    def uniform(cls, shape, ranks):
        """Create a density instance which samples indices uniformly in each
        dimension.
        """
        train = ones(shape, ranks)
        return cls.from_train(train)
