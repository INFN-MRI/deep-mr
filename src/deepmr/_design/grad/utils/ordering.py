"""View ordering utils."""

__all__ = ["get_centerout_order"]

import numpy as np


def get_centerout_order(v):
    n = len(v)

    vco = np.zeros_like(v)
    for i in range(int(n // 2)):
        vco[2 * i] = n // 2 + i
        vco[2 * i + 1] = n // 2 - i - 1

    if n % 2 != 0:
        vco[-1] = n - 1

    return v[vco.astype(int)]
