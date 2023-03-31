import numpy as np
from numpy.testing import assert_equal
from protes.protes import minimize


def test_minimize():
    (x, y), _ = minimize(
        fn=lambda ix: ix.sum(axis=1),
        shape=(5, ) * 10,
        max_trials=5000,
    )
    assert_equal(x, np.zeros_like(x))
    assert_equal(y.item(), 0)
