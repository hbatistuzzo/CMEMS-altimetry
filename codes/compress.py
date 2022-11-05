#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:53:22 2020

@author: hbatistuzzo
"""

import numpy as np
import netCDF4 as nc
import xarray as xr
import multiprocessing as mp
import os
import pickle
import time
import humanize
from sys import getsizeof
import time
from dask.diagnostics import ProgressBar
from collections import OrderedDict as od
print("imported packages...")

# ==================================================================
# Making a proper netcdf here with 12 variables: mean and adt for cmems vars
ldir0 = r"/media/hbatistuzzo/DATA/alt_cmems/"    

infile = open(ldir0+'adt_full_mean_and_std.pckl', 'rb')
adt = pickle.load(infile)
infile.close()
adt_mean = adt[0]
adt_std = adt[1]

# adt_mean.to_netcdf(ldir0+'adt_full_mean.nc')
# adt_std.to_netcdf(ldir0+'adt_full_std.nc')

infile = open(ldir0+'sla_full_mean_and_std.pckl', 'rb')
sla = pickle.load(infile)
infile.close()
sla_mean = sla[0]
sla_std = sla[1]

# sla_mean.to_netcdf(ldir0+'sla_full_mean.nc')
# sla_std.to_netcdf(ldir0+'sla_full_std.nc')

infile = open(ldir0+'ugos_full_mean_and_std.pckl', 'rb')
ugos = pickle.load(infile)
infile.close()
ugos_mean = ugos[0]
ugos_std = ugos[1]

# ugos_mean.to_netcdf(ldir0+'ugos_full_mean.nc')
# ugos_std.to_netcdf(ldir0+'ugos_full_std.nc')

infile = open(ldir0+'vgos_full_mean_and_std.pckl', 'rb')
vgos = pickle.load(infile)
infile.close()
vgos_mean = vgos[0]
vgos_std = vgos[1]

# vgos_mean.to_netcdf(ldir0+'vgos_full_mean.nc')
# vgos_std.to_netcdf(ldir0+'vgos_full_std.nc')

infile = open(ldir0+'ugosa_full_mean_and_std.pckl', 'rb')
ugosa = pickle.load(infile)
infile.close()
ugosa_mean = ugosa[0]
ugosa_std = ugosa[1]

# ugosa_mean.to_netcdf(ldir0+'ugosa_full_mean.nc')
# ugosa_std.to_netcdf(ldir0+'ugosa_full_std.nc')

infile = open(ldir0+'vgosa_full_mean_and_std.pckl', 'rb')
vgosa = pickle.load(infile)
infile.close()
vgosa_mean = vgosa[0]
vgosa_std = vgosa[1]

# vgosa_mean.to_netcdf(ldir0+'vgosa_full_mean.nc')
# vgosa_std.to_netcdf(ldir0+'vgosa_full_std.nc')



# One .nc for means, one for stds
lat = adt_mean.latitude.values
lon = adt_mean.longitude.values

ddd = {'lat': {'dims': 'lat','data': lat, 'attrs': {'units': 'deg N/S'}},
       'lon': {'dims': 'lon', 'data': lon, 'attrs': {'units': 'deg E/W'}}}

z_c1 = od()
z_c2 = od()
z_c3 = od()
z_c4 = od()
z_c1['adt_mean'] = adt_mean.values
z_c1['sla_mean'] = sla_mean.values
z_c2['ugos_mean'] = ugos_mean.values
z_c2['vgos_mean'] = vgos_mean.values
z_c2['ugosa_mean'] = ugosa_mean.values
z_c2['vgosa_mean'] = vgosa_mean.values
z_c3['adt_std'] = adt_std.values
z_c3['sla_std'] = sla_std.values
z_c4['ugos_std'] = ugos_std.values
z_c4['vgos_std'] = vgos_std.values
z_c4['ugosa_std'] = ugosa_std.values
z_c4['vgosa_std'] = vgosa_std.values

#loop for adt and sla. Converting to... int16? what about float64?
encoding = {}
for key in z_c1.keys():
    ddd.update({key: {'dims': ('lat', 'lon'), 'data': z_c1[key],
                          'attrs': {'units': 'm'}}})
    encoding.update({key: {'dtype': 'int16', 'scale_factor': 0.01,
                                'zlib': True, '_FillValue': -9999999}})

for key in z_c2.keys():
    ddd.update({key: {'dims': ('lat', 'lon'), 'data': z_c2[key],
                          'attrs': {'units': 'm/s'}}})
    encoding.update({key: {'dtype': 'int16', 'scale_factor': 0.01,
                                'zlib': True, '_FillValue': -9999999}})
for key in z_c3.keys():
    ddd.update({key: {'dims': ('lat', 'lon'), 'data': z_c3[key],
                          'attrs': {'units': 'm'}}})
    encoding.update({key: {'dtype': 'int16', 'scale_factor': 0.01,
                                'zlib': True, '_FillValue': -9999999}})

for key in z_c4.keys():
    ddd.update({key: {'dims': ('lat', 'lon'), 'data': z_c4[key],
                          'attrs': {'units': 'm/s'}}})
    encoding.update({key: {'dtype': 'int16', 'scale_factor': 0.01,
                                'zlib': True, '_FillValue': -9999999}})
ds = xr.Dataset.from_dict(ddd)
ds.to_netcdf('/media/hbatistuzzo/DATA/alt_cmems/full_means_stds.nc', format='NETCDF4',
             encoding=encoding)
ds.to_netcdf('/media/hbatistuzzo/DATA/alt_cmems/full_means_stds.nc', format='NETCDF4')



test = xr.open_dataset(ldir0 + '1993/01/dt_global_allsat_phy_l4_19930101_20190101.nc') #example

###############################################################

ldir0 = r"/media/hbatistuzzo/DATA/alt_cmems/"  

ds = xr.open_dataset(ldir0 + 'full_means_stds.nc') #to open the means and stds



#now for the monthly data
ldir1 = r"/media/hbatistuzzo/DATA/alt_cmems/processed/monthly_timeseries/"

#let's open the ADT mean monthly set
infile = open(ldir1 + 'adt_mean_months.pckl', 'rb')
adt_mean_months = pickle.load(infile) #is a DataArray 12 x 720 x 1440 #WHOOPS fix this
infile.close()

infile = open(ldir1 + 'adt_mean_months.pckl', 'rb')
adt_mean_months = pickle.load(infile) #is a DataArray 12 x 720 x 1440 #WHOOPS fix this
infile.close()

infile = open(ldir1 + 'adt_mean_months.pckl', 'rb')
adt_mean_months = pickle.load(infile) #is a DataArray 12 x 720 x 1440 #WHOOPS fix this
infile.close()

infile = open(ldir1 + 'adt_mean_months.pckl', 'rb')
adt_mean_months = pickle.load(infile) #is a DataArray 12 x 720 x 1440 #WHOOPS fix this
infile.close()

infile = open(ldir1 + 'adt_mean_months.pckl', 'rb')
adt_mean_months = pickle.load(infile) #is a DataArray 12 x 720 x 1440 #WHOOPS fix this
infile.close()

infile = open(ldir1 + 'adt_mean_months.pckl', 'rb')
adt_mean_months = pickle.load(infile) #is a DataArray 12 x 720 x 1440 #WHOOPS fix this
infile.close()

# One .nc for means, one for stds
lat = adt_mean_months.latitude.values
lon = adt_mean_months.longitude.values

ddd = {'lat': {'dims': 'lat','data': lat, 'attrs': {'units': 'deg N/S'}},
       'lon': {'dims': 'lon', 'data': lon, 'attrs': {'units': 'deg E/W'}}}

z_c1 = od()
z_c2 = od()
z_c3 = od()
z_c4 = od()
z_c1['adt_mean'] = adt_mean.values
z_c1['sla_mean'] = sla_mean.values
z_c2['ugos_mean'] = ugos_mean.values
z_c2['vgos_mean'] = vgos_mean.values
z_c2['ugosa_mean'] = ugosa_mean.values
z_c2['vgosa_mean'] = vgosa_mean.values
z_c3['adt_std'] = adt_std.values
z_c3['sla_std'] = sla_std.values
z_c4['ugos_std'] = ugos_std.values
z_c4['vgos_std'] = vgos_std.values
z_c4['ugosa_std'] = ugosa_std.values
z_c4['vgosa_std'] = vgosa_std.values


###############################################################################
infile = open(ldir0+'cube.pckl', 'rb')
cube = pickle.load(infile)
infile.close()

lat1=cube.adt.sel(latitude=-0.875)

with ProgressBar():
    lat1.to_netcdf('/media/hbatistuzzo/DATA/alt_cmems/lat_test2.nc', format='NETCDF4',
             encoding={'adt':{'dtype': 'int16', 'scale_factor': 0.01,'zlib': True, '_FillValue': -9999999}})

# lat1=cube.sel(latitude=-89.875)
# with ProgressBar():
#     lat1.to_netcdf('/media/hbatistuzzo/DATA/alt_cmems/lat_test2.nc', format='NETCDF4',
#              encoding={'crs':{'dtype': 'int16', 'scale_factor': 0.01,'zlib': True, '_FillValue': -9999999},
#              'lat_bnds':{'dtype': 'int16', 'scale_factor': 0.01,'zlib': True, '_FillValue': -9999999},
#              'lon_bnds':{'dtype': 'int16', 'scale_factor': 0.01,'zlib': True, '_FillValue': -9999999},
#              'err':{'dtype': 'int16', 'scale_factor': 0.01,'zlib': True, '_FillValue': -9999999},
#              'adt':{'dtype': 'int16', 'scale_factor': 0.01,'zlib': True, '_FillValue': -9999999},
#              'ugos':{'dtype': 'int16', 'scale_factor': 0.01,'zlib': True, '_FillValue': -9999999},
#              'vgos':{'dtype': 'int16', 'scale_factor': 0.01,'zlib': True, '_FillValue': -9999999},
#              'sla':{'dtype': 'int16', 'scale_factor': 0.01,'zlib': True, '_FillValue': -9999999},
#              'ugosa':{'dtype': 'int16', 'scale_factor': 0.01,'zlib': True, '_FillValue': -9999999},
#              'vgosa':{'dtype': 'int16', 'scale_factor': 0.01,'zlib': True, '_FillValue': -9999999}})


test.to_netcdf(ldir1+'test.nc',encoding={'adt':{'dtype': 'int32', 'scale_factor': 0.1, '_FillValue': -9999,'zlib': True,'complevel': 1}})



test = xr.open_dataset('/media/hbatistuzzo/DATA/alt_cmems/lat_test.nc') #example





















