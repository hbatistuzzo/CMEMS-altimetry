#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 09:38:13 2020

@author: paulo
"""

import numpy as np
from fir import gauss, obconvolve, maxvar, varexp, f_longterm, f_annual,\
                f_wave
import matplotlib.pyplot as plt
from matplotlib import rc
rc('image', cmap='RdBu_r')
# data resoltion in days and degrees of longitude
dt = 1
dx = 0.25

# dimensions of the synthetic altimeter data in days and degrees
nt = 2*365    # 5 years
nx = 71      # 71 degrees

# -----------------------------------------------------------------------------
# Build synthetic signals
# period in days
T = 183
# wavelength in kmimport numpy as np
from numpy import sin, cos, matrix, pi, arctan2, arange
from numpy.linalg import lstsq, norm
L = -620
# local latitude
lat = -18.625
# amplitudes of the synthetic signals
# (trend, annual, sinusoidal wave, random noise)
At, Aa, As, An = [10, 10, 10, 10]
# simulate grid resolution of 1 day and 0.25°
t1 = dt*np.arange(0, np.round(nt/dt)+1).astype(float)
x1 = dx*np.arange(np.round(nx/dx)).astype(float)
x2, t2, = np.meshgrid(x1, t1)
# trend
zt = At*((t2-t2.mean())/t2.mean())
# annual
za = Aa*np.sin((2*np.pi/365.25)*t2)
# wave
# convert L to degrees
Lo = L/(111.195*np.cos(lat*np.pi/180))
zs = As*np.sin((2*np.pi/Lo)*x2 - (2*np.pi/T)*t2)
# noise
zn = np.random.random(x2.shape)
g, _, _ = gauss(T//4, T//4, 2)
zn = obconvolve(zn, g)
zn = (An/np.sqrt(2))*(zn-zn.mean())/zn.std()
# assemble synthetic signal
z = zt + za + zs + zn
# -----------------------------------------------------------------------------
# And now for something completely different...
# -----------------------------------------------------------------------------
z_ori = z.copy()
z_trend = f_longterm(z, ny=3, dt=dt)
z_trend = maxvar(z_trend, z)*z_trend
z = z_ori - z_trend

z_annual = f_annual(z, dt)
z_annual = maxvar(z_annual, z)*z_annual
z = z - z_annual

z_12 = f_wave(z, dx, dt, lat, L, T, 1)
z_12 = maxvar(z_12, z)*z_12
z = z - z_12

plt.figure(0)
plt.clf()
plt.pcolormesh(x1, t1, z_ori)
plt.title('z original')
plt.colorbar()

plt.figure(1)
plt.clf()
plt.pcolormesh(x1, t1, z_trend)
plt.title('z trend, v={:4.1f}%'.format(100*varexp(z_ori, z_trend)))
plt.colorbar()

plt.figure(2)
plt.clf()
plt.pcolormesh(x1, t1, z_annual)
plt.title('z annual, v={:4.1f}%'.format(100*varexp(z_ori, z_annual)))
plt.colorbar()

plt.figure(3)
plt.clf()
plt.pcolormesh(x1, t1, z_12)
plt.title('z 12, v={:4.1f}%'.format(100*varexp(z_ori, z_12)))
plt.colorbar()

plt.figure(4)
plt.clf()
plt.pcolormesh(x1, t1, z)
plt.title('z residual, v={:4.1f}%'.format(100*varexp(z_ori, z)))
plt.colorbar()
