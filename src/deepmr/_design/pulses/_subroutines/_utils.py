"""Utils for RF pulse design."""

__all__ = ['dinf', 'leja']

import numpy as np

def dinf(d1=0.01, d2=0.01):
    """Calculate D infinity for a linear phase filter.

    Args:
        d1 (float): passband ripple level in M0**-1.
        d2 (float): stopband ripple level in M0**-1.

    Returns:
        (float): D infinity.

    References:
        Pauly J, Le Roux P, Nishimra D, Macovski A. Parameter relations for the
        Shinnar-Le Roux selective excitation pulse design algorithm.
        IEEE Tr Medical Imaging 1991; 10(1):53-65.
    """
    a1 = 5.309e-3
    a2 = 7.114e-2
    a3 = -4.761e-1
    a4 = -2.66e-3
    a5 = -5.941e-1
    a6 = -4.278e-1

    l10d1 = np.log10(d1)
    l10d2 = np.log10(d2)

    d = (a1 * l10d1 * l10d1 + a2 * l10d1 + a3) * l10d2 + (a4 * l10d1 * l10d1 + a5 * l10d1 + a6)

    return d


def leja(x):
    """ Perform leja ordering of roots of a polynomial.
    
    Orders roots in a way suitable to accurately compute polynomial
    coefficients.
    
    Args:
        x (array): roots to be ordered.
        
    Returns:
        (array) ordered roots.
        
    References:
        Lang, M. and B. Frenzel. 1993.
        A New and Efficient Program for Finding All Polynomial Roots. Rice
        University ECE Technical Report, no. TR93-08, 1993.
    """
    n = np.size(x)
    # duplicate roots to n+1 rows
    a = np.tile(np.reshape(x, (1, n)), (n+1, 1))
    # take abs of first row
    a[0, :] = np.abs(a[0, :])

    tmp = np.zeros(n+1, dtype=complex)

    # find index of max abs value
    ind = np.argmax(a[0, :])
    if ind != 0:
        tmp[:] = a[:, 0]
        a[:, 0] = a[:, ind]
        a[:, ind] = tmp

    x_out = np.zeros(n, dtype=complex)
    x_out[0] = a[n-1, 0]  # first entry of last row
    a[1, 1:] = np.abs(a[1, 1:] - x_out[0])

    foo = a[0, 0:n]

    for l in range(1, n-1):
        foo = np.multiply(foo, a[l, :])
        ind = np.argmax(foo[l:])
        ind = ind + l
        if l != ind:
            tmp[:] = a[:, l]
            a[:, l] = a[:, ind]
            a[:, ind] = tmp
            # also swap inds in foo
            tmp[0] = foo[l]
            foo[l] = foo[ind]
            foo[ind] = tmp[0]
        x_out[l] = a[n-1, l]
        a[l+1, (l+1):n] = np.abs(a[l+1, (l+1):] - x_out[l])

    x_out = a[n, :]

    return x_out

