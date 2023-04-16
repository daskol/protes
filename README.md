# PROTES

*PRobability Optimizer with TEnsor Sampling*

## Overview

This is an implementation of PROTES probabilistic optimization algorithm
[[1][1]] from scratch. Its original implementation is available at
[anabatsh/protes][2] and published implementation is at [anabatsh/TT_pro][3].
The package can be directly installed from GitHub as follows.

```shell
pip install git+https://github.com/daskol/protes.git
```

As an example, let's solve a toy problem. Consider a minimization problem of
summation 10 non-negative numbers, not greater than 5. Then this problem can be
easily solved with PROTES.

```python
from protes import minimize

(x, y), pdf = minimize(
    fn=lambda ix: ix.sum(axis=1),
    shape=(5, ) * 10,
    max_trials=5000,
)

print('x =', x)  # x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
print('y =', y)  # y = 0
```

All objects provided `protes` are actual `pytree`'s and could be used with
`jax.jit`. So, it is possible to pickle and unpickle them. With the snippet
above, one can easily save density estimator `pdf` to and load it from a file.

```python
from pickle import dump, load

with open('pdf.pkl', 'wb') as fout:
    dump(pdf, fout)

with open('pdf.pkl', 'rb') as fin:
    pdf = load(fin)

key = jax.random.PRNGKey(42)
batch = pdf.sample(key, (2, 1))
assert batch.ndim == 3
assert batch.shape = (2, 1, 10)
```

Also, the package provides an implementation of PROTES as a stateful object
`TensorTrainSampler` for optimization in the manner of active learning. Again,
everything can be `jax.jit`'ed.

```python
from functools import partial
from protes import TensorTrainSampler
import jax


@partial(jax.jit, static_argnames=('fn', ))
def step(key, sampler, fn):
    indices = sampler.sample(key, (2, ))
    values = fn(indices)
    sampler, _ = sampler.submit(indices, values)
    return sampler

# Instantiate sampler, make a step, and save sampler.
key = jax.random.PRNGKey(42)
sampler = TensorTrainSampler((5, ) * 10)
sampler = step(key, sampler, lambda ix: ix.sum(axis=1))
sampler.save('sampler.pkl')

# Load sampler and continue optimization.
sampler = TensorTrainSampler.load('sampler.pkl')
sampler = step(key, sampler, lambda ix: ix.sum(axis=1))
sampler.save('sampler.pkl')
```

Note, `optax` implements optimizers as closures. This results in inability to
pickle it with `pickle` in standard library. Hopefully, there is a
`cloudpickle` which is able to serialize and deserialize closures and lambdas.
So, `TensorTrainSampler` has methods `load` and `save` in order to hide these
details. Other objects are picklable with standard one.

## Citation

```bibtex
@article{batsheva2023protes,
    author    = {Batsheva, Anastasia and Chertkov, Andrei  and Ryzhakov, Gleb and Oseledets, Ivan},
    year      = {2023},
    title     = {PROTES: Probabilistic Optimization with Tensor Sampling},
    journal   = {arXiv preprint arXiv:2301.12162},
    url       = {https://arxiv.org/pdf/2301.12162.pdf}
}
```

[1]: https://arxiv.org/abs/2301.12162
[2]: https://github.com/anabatsh/protes
[3]: https://github.com/anabatsh/TT_pro
