#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 10:07:55 2020

@author: paulo
"""

from numpy import sin, cos, matrix, pi, arctan2, arange
from numpy.linalg import lstsq, norm
from numpy.random import random
import matplotlib.pyplot as plt


def sinfit(t, z, w):
    '''
    if t is time and z is the independent variable, w is the frequency to be
    used in fitting [A*sin(w*t + phi) + a*t + b] to z. w=2*pi/P
    returns (A, phi, a, b)
    '''
    B = matrix(z).T
    rows = [[sin(w*t), cos(w*t), t, 1] for t in t]
    A = matrix(rows)
    (w, residuals, rank, sing_vals) = lstsq(A, B, rcond=None)
    phi = arctan2(w[1, 0], w[0, 0])
    A = norm([w[0, 0], w[1, 0]], 2)
    a = w[2, 0]
    b = w[3, 0]
    return (A, phi, a, b)


t = arange(1000)
w = 2*pi/100
phi = 27
A = 12
An = 6
a = 0.01
b = -2
zs = A*sin(w*t + phi) + a*t + b
zn = An*random(zs.shape)
z = zs + zn

(Af, phif, af, bf) = sinfit(t, z, w)
zf = Af*sin(w*t + phif) + af*t + bf

plt.clf()
plt.plot(t, z)
plt.plot(t, zf)
