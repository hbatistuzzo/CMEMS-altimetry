#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 10:56:54 2020

@author: paulo
"""

import numpy as np
import xarray as xr
from collections import OrderedDict as od

# data resoltion in days and degrees of longitude
delt = 1
delx = 0.25

# dimensions of the synthetic altimeter data in days and degrees
nt = 2*365    # 5 years
nx = 71      # 71 degrees

# -----------------------------------------------------------------------------
# Build synthetic signals
# period in days
T = 183 #semi-annual
# wavelength in km
L = -620
# local latitude
lat = -18.625
# amplitudes of the synthetic signals
# (trend, annual, sinusoidal wave, random noise)
At, Aa, As, An = [10, 10, 10, 10]
# simulate grid resolution of 1 day and 0.25°
t1 = delt*np.arange(0, np.round(nt/delt)+1).astype(float)
x1 = delx*np.arange(np.round(nx/delx)).astype(float)
x2, t2, = np.meshgrid(x1, t1)
# build dictionary of components
z_c = od()
# trend
z_c['trend'] = At*((t2-t2.mean())/t2.mean())
# annual
z_c['annual'] = Aa*np.sin((2*np.pi/365.25)*t2)
# wave
# convert L to degrees
Lo = L/(111.195*np.cos(lat*np.pi/180))
z_c['R_183'] = As*np.sin((2*np.pi/Lo)*x2 - (2*np.pi/T)*t2)

dd = {'time': {'dims': 'time', 'data': t1, 'attrs': {'units': 'd'}},
      'lon': {'dims': 'lon', 'data': x1, 'attrs': {'units': 'deg E/W'}}}
encoding = {}
for key in z_c.keys():
    dd.update({'z_'+key: {'dims': ('time', 'lon'), 'data': z_c[key],
                          'attrs': {'units': 'mm'}}})
    encoding.update({'z_'+key: {'dtype': 'int16', 'scale_factor': 0.01,
                                'zlib': True, '_FillValue': -9999999}})
ds = xr.Dataset.from_dict(dd)
ds.to_netcdf('/media/hbatistuzzo/DATA/alt_cmems/nc0.nc', format='NETCDF4',
             encoding=encoding)
ds.to_netcdf('/media/hbatistuzzo/DATA/alt_cmems/nc1.nc', format='NETCDF4')

ldir0=r"/media/hbatistuzzo/DATA/alt_cmems/"    
ayy = xr.open_dataset(ldir0 + 'nc0.nc')

