# https://arxiv.org/pdf/1512.04442.pdf
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


def fit_costhetal(x, *coeffs):
    # Fit the function
    summation = 0
    for i in range(0, 5):
        p = np.polynomial.Legendre.basis(i).convert(kind=np.polynomial.Polynomial)
        summation += p(x) * coeffs[i]
    return summation


def fit_costhetal_q2(xy, *coeffs):
    # Fit the function
    summation = 0
    x = xy[0]
    y = xy[1]
    print(coeffs)
    for i in range(0, 5):
        pcl = np.polynomial.Legendre.basis(i).convert(kind=np.polynomial.Polynomial)
        for j in range(0, 6):
                pq2 = np.polynomial.Legendre.basis(j).convert(kind=np.polynomial.Polynomial)
                summation += pcl(x) * pq2(y) * coeffs[i*5 + j]
    return summation


def accept_costhetal(accept_data_range):
    # Generate binned data
    n, bins, patches = plt.hist(accept_data_range["costhetal"], bins=25,
                                histtype=u'step', density=True)

    bins = bins[1:]
    # Acceptance function
    coeffs = [1] * 5
    o = scipy.optimize.curve_fit(fit_costhetal, bins, n, coeffs)
    # plt.scatter(n, bins)
    eff = fit_costhetal(bins, *o[0])
    plt.scatter(bins, eff)
    plt.show()
    a = 0


def fitness(params):
    # Minimising the sum of squares of the difference between the model and the data
    a, b, c, d = params
    Xs = odeint(self.lv_derivative, init_cond, t, args=(a, b, c, d))
    # ss = lambda data, model: ((data - model) ** 2).sum()
    return ss(Xs[:, 0], x) / max(x) + ss(Xs[:, 1], y) / max(y)


def ss(data, model):
    adjust = []
    for i in range(len(data)):
        adjust.append((data[i] - model[i]) ** (2))
    return sum(adjust)


def accept_costhetal_q2(accept_data_range):
    # Generate binned data
    # fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    # ax[0].hist(accept_data_range["costhetal"], bins=25,
    #                             histtype=u'step', density=True)
    # ax[1].hist(accept_data_range["q2"], bins=25,
    #                             histtype=u'step', density=True)

    # H, edges = np.histogramdd([accept_data_range["costhetal"], accept_data_range["q2"]], bins=(25, 25))
    fig = plt.figure()
    ax = fig.add_subplot()
    hist, xedges, yedges = np.histogram2d(accept_data_range["costhetal"], accept_data_range["q2"], bins=7)
    # Acceptance function
    # coeffs = [1] * 5
    coeffs = np.ones((5, 6))
    o = scipy.optimize.curve_fit(fit_costhetal_q2, (xedges, yedges), hist.ravel(), coeffs)
    # plt.scatter(n, bins)
    eff = fit_costhetal_q2((xedges, yedges), *o[0])
    # ax[0].scatter(bins, eff)
    # ax[1]
    plt.show()
    a = 0
