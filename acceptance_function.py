# https://arxiv.org/pdf/1512.04442.pdf
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


def fit_costhetal(x, *coeffs):
    # Fit the function
    summation = 0
    for i in range(5):
        p = np.polynomial.Legendre.basis(i).convert(kind=np.polynomial.Polynomial)
        summation += p(x) * coeffs[i]
    return summation


def accept_costhetal(accept_data_range):
    # Generate binned data
    n, bins, patches = plt.hist(accept_data_range["costhetal"], bins=25)

    # Acceptance function
    coeffs = [1] * 5
    o = scipy.optimize.curve_fit(fit_costhetal, bins, n, coeffs)
    a = 0