#!/usr/bin/env python3

import numpy as np


def lorenzian(x, x0, gamma, A):
    return A/np.pi* (0.5*gamma)/((x-x0)**2+(0.5*gamma)**2)


def gaussian(p, x):

    x0, sigma, A = p

    return (A/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x-x0)/sigma)**2)
