import numpy as np

from tfm_volatility.utils.seeds import set_global_seed


def test_set_global_seed_makes_numpy_reproducible():
    set_global_seed(42)
    a = np.random.rand(5)
    set_global_seed(42)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_set_global_seed_makes_python_random_reproducible():
    import random
    set_global_seed(42)
    a = [random.random() for _ in range(5)]
    set_global_seed(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_global_seed_returns_the_seed_used():
    used = set_global_seed(1337)
    assert used == 1337
