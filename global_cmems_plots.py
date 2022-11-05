#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Jan  1 13:10:54 2021

@author: hbatistuzzo
"""

# Featuring functions and a modular structure.
# CMEMS data doesnt start at greenwich so we need shiftgrid

from pylab import text
import numpy as np
import netCDF4 as nc
import xarray as xr
import os
import pickle
import time
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy.ma as ma
import warnings
import matplotlib.cbook
import cartopy
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
import cmocean
from dask.diagnostics import ProgressBar
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker

def stats(vari):
    mu = np.nanmean(vari)
    sigma = np.nanstd(vari)
    vari_min = np.nanmin(vari)
    vari_max = np.nanmax(vari)
    print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {vari_min:.2f}, max is {vari_max:.2f}')
    return mu, sigma, vari_min, vari_max


def shiftgrid(lon0,datain,lonsin,start=True,cyclic=360.0):
    """
    Shift global lat/lon grid east or west.
    .. tabularcolumns:: |l|L|
    ==============   ====================================================
    Arguments        Description
    ==============   ====================================================
    lon0             starting longitude for shifted grid
                     (ending longitude if start=False). lon0 must be on
                     input grid (within the range of lonsin).
    datain           original data with longitude the right-most
                     dimension.
    lonsin           original longitudes.
    ==============   ====================================================
    .. tabularcolumns:: |l|L|
    ==============   ====================================================
    Keywords         Description
    ==============   ====================================================
    start            if True, lon0 represents the starting longitude
                     of the new grid. if False, lon0 is the ending
                     longitude. Default True.
    cyclic           width of periodic domain (default 360)
    ==============   ====================================================
    returns ``dataout,lonsout`` (data and longitudes on shifted grid).
    """
    if np.fabs(lonsin[-1]-lonsin[0]-cyclic) > 1.e-4:
        # Use all data instead of raise ValueError, 'cyclic point not included'
        start_idx = 0
    else:
        # If cyclic, remove the duplicate point
        start_idx = 1
    if lon0 < lonsin[0] or lon0 > lonsin[-1]:
        raise ValueError('lon0 outside of range of lonsin')
    i0 = np.argmin(np.fabs(lonsin-lon0))
    i0_shift = len(lonsin)-i0
    if ma.isMA(datain):
        dataout  = ma.zeros(datain.shape,datain.dtype)
    else:
        dataout  = np.zeros(datain.shape,datain.dtype)
    if ma.isMA(lonsin):
        lonsout = ma.zeros(lonsin.shape,lonsin.dtype)
    else:
        lonsout = np.zeros(lonsin.shape,lonsin.dtype)
    if start:
        lonsout[0:i0_shift] = lonsin[i0:]
    else:
        lonsout[0:i0_shift] = lonsin[i0:]-cyclic
    dataout[...,0:i0_shift] = datain[...,i0:]
    if start:
        lonsout[i0_shift:] = lonsin[start_idx:i0+start_idx]+cyclic
    else:
        lonsout[i0_shift:] = lonsin[start_idx:i0+start_idx]
    dataout[...,i0_shift:] = datain[...,start_idx:i0+start_idx]
    return dataout,lonsout

print("imported packages...")


###################################################################################################
# 1) CMEMS FULL SERIES PLOTS
ldir0 = r"/media/hbatistuzzo/DATA/Ilhas/CMEMS/alt_cmems/processed/full_timeseries/"
path = 'full_means_stds.nc'
ds = xr.open_dataset(ldir0+path) # Means and STDs for ADT, SLA, UGOS, VGOS, UGOSA, VGOSA


# 2) Get some stats:
def stats(vari):
    mu = np.nanmean(vari.values)
    sigma = np.nanstd(vari.values)
    vari_min = np.nanmin(vari.values)
    vari_max = np.nanmax(vari.values)
    print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {vari_min:.2f}, max is {vari_max:.2f}')
    return mu, sigma, vari_min, vari_max

vari = ds.vgos_mean
variv = ds.vgos_mean.values # 720 x 1440 array float 32
mu,sigma,vari_min,vari_max=stats(vari)

# 3) Set colorbar intervals
[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

# 4) aux
lon = vari.lon.values # 1440 array float32
lat = vari.lat.values # 720 ~~~
ticks_var = [ssm,sm,m,ms,mss]
gfont = {'fontname':'Helvetica','fontsize' : 20}

# 5) Wrap greenwich if necessary
variv, lon = shiftgrid(180., variv, lon, start=False) #0 to 360 needs wrapping


# 6) Plot
fig = plt.figure(figsize=(8,6),dpi=300)
ax = plt.axes(projection=ccrs.Robinson())

# CS = plt.contour(lon, lat, variv, transform=ccrs.PlateCarree(),levels=[ssm,sm,m,ms,mss],
#                   linewidths=0.5,colors='k',zorder=1,inline=True)
# fmt = {} 
# strs = ['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']
# for l, s in zip(CS.levels, strs): 
#     fmt[l] = s
# ax.clabel(CS, CS.levels, inline=True,fontsize=5,fmt=fmt)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1,zorder=2)
ax.add_feature(cfeature.LAND, zorder=3,edgecolor='black',linewidths=0.25)
ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1,zorder=4)
ax.add_feature(cfeature.BORDERS, linewidths=0.25,zorder=5)
ax.add_feature(cfeature.RIVERS, linewidths=0.25,zorder=6)
ax.set_global()
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
plt.title("CMEMS VGOS 1993-2019 mean",fontdict = gfont,pad=10)
cf = plt.pcolormesh(lon,lat,variv,transform=ccrs.PlateCarree(),cmap=cmocean.cm.balance,
                    vmin=ssm,vmax=mss,zorder=0)
gl = ax.gridlines(draw_labels=True,xlocs=np.arange(-120, 121, 60),
                  ylocs=np.arange(-90, 91, 30),
                  linewidth=0.25,color='black',zorder = 7)
gl.xlabel_style = {'size': 10, 'color': 'black'}
gl.ylabel_style = {'size': 10, 'color': 'black'}
gl.rotate_labels = False
gl.ypadding = 30
gl.xpadding = 10
cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_var,pad=0.1,extend='both')
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_var,extend='both')
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.set_xticklabels(['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$',
                          '$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
lon_formatter = LongitudeFormatter(zero_direction_label=True)
# cbar.ax.text(ssm-abs((0.2*ssm)),0,f'MIN\n{vari_min:.2f}',ha='center',va='center')
# cbar.ax.text(mss+abs((0.2*ssm)),0,f'MAX\n{vari_max:.2f}',ha='center',va='center')
cbar.set_label('Meridional Geostrophic Velocity (m/s)')
cbar.ax.get_yaxis().set_ticks([])
# cbar.ax.text(ssm-0.1*ssm,0.3,f'MIN\n{vari_min:.2f}',ha='center',va='center')
# cbar.ax.text(mss+0.1*ssm,0.3,f'MAX\n{vari_max:.2f}',ha='center',va='center')
ax.set_aspect('auto')
text(0, 0, f'MIN = {vari_min:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
text(1, 0, f'MAX = {vari_max:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
if not (os.path.exists(ldir0 + '/global/vgos_full_mean_global.png')):
    plt.savefig(ldir0 + '/global/vgos_full_mean_global.png',bbox_inches='tight')
plt.show()


###################################################################################################
# 1) CMEMS MONTHLY SERIES PLOTS
# from dask.diagnostics import ProgressBar
# from dask.distributed import Client
# #lets bring up the dask monitor. http://localhost:8787/status
# # client = Client()
# client = Client() #ok lets try this
# client

ldir0=r"/media/hbatistuzzo/DATA/Ilhas/CMEMS/alt_cmems/processed/monthly_timeseries/"
ds = xr.open_dataset(ldir0+'monthly_means_stds.nc')

month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

#plotting monthly means
lon = ds.lon.values
lat = ds.lat.values


adt_mean = ds.adt_mean_months
sla_mean = ds.sla_mean_months
ugos_mean = ds.ugos_mean_months
vgos_mean = ds.vgos_mean_months



var = {}
namespace = globals()
var_list=[]
for m in np.arange(0,12):
    var[month[m]] = vgos_mean[m,:,:].values
    var_list.append(vgos_mean[m,:,:].values) #this works for separating the months
    namespace[f'vgos_mean_{month[m]}'] = vgos_mean[m] #separates the 12 dataarrays by name
    print('this worked')


#for a fixed colorbar, lets get the mean of the means and the mean of the stds
mu = round(np.nanmean(vgos_mean.values),4)
sigma = round(np.nanstd(vgos_mean.values),4)
z_min = round(np.nanmin(vgos_mean.values),4)
z_max = round(np.nanmax(vgos_mean.values),4)
print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {z_min:.2f}, max is {z_max:.2f}')

[ssm_all,sm_all,m_all,ms_all,mss_all] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
ticks_var = [ssm_all,sm_all,m_all,ms_all,mss_all]
gfont = {'fontname':'Helvetica','fontsize' : 20}


#### RIGHT, we need to take the mean of the means..

figs = []
for mon in var.keys():
    lon = ds.lon.values
    # 5) Wrap greenwich
    var[mon], lon = shiftgrid(180., var[mon], lon, start=False) #0 to 360 needs wrapping
    mu = round(np.nanmean(var[mon]),2)
    sigma = round(np.nanstd(var[mon]),2)
    vari_min = round(np.nanmin(var[mon]),2)
    vari_max = round(np.nanmax(var[mon]),2)
    print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {vari_min:.2f}, max is {vari_max:.2f}')
    [ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
              np.around(mu-sigma,decimals=2),
              np.around(mu,decimals=2),
              np.around(mu+sigma,decimals=2),
              np.around(mu+2*sigma,decimals=2)]
    print([ssm,sm,m,ms,mss])
    lons = vgos_mean.lon.values
    fig = plt.figure(figsize=(8,6),dpi=300)
    ax = plt.axes(projection=ccrs.Robinson())
    # CS = plt.contour(lon, lat, var[mon], transform=ccrs.PlateCarree(),levels=[ssm,sm,m,ms,mss],
    #                   linewidths=0.5,colors='k',zorder=1,inline=True)
    # fmt = {} 
    # strs = ['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']
    # for l, s in zip(CS.levels, strs): 
    #         fmt[l] = s
    # ax.clabel(CS, CS.levels, inline=True,fontsize=5,fmt=fmt)
    ax.coastlines(resolution='50m', color='black', linewidth=0.25)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1,zorder=2)
    ax.add_feature(cfeature.LAND, zorder=3,edgecolor='black',linewidths=0.25)
    ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1,zorder=4)
    ax.add_feature(cfeature.BORDERS, linewidths=0.25,zorder=5)
    ax.add_feature(cfeature.RIVERS, linewidths=0.25,zorder=6)
    ax.set_global()
    states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
    ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
    plt.title(f"CMEMS VGOS 1993-2019 {mon} mean",fontdict = gfont,pad=10)
    cf = plt.pcolormesh(lon,lat,var[mon],transform=ccrs.PlateCarree(),cmap=cmocean.cm.balance,
                    vmin=ssm_all,vmax=mss_all,zorder=0)
    gl = ax.gridlines(draw_labels=True,xlocs=np.arange(-120, 121, 60),
                  ylocs=np.arange(-90, 91, 30),
                  linewidth=0.25,color='black',zorder = 7)
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}
    gl.rotate_labels = False
    gl.ypadding = 30
    gl.xpadding = 10
    gl.xpadding = 10
    cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_var,pad=0.1,extend='both')
    pos = cbar.ax.get_position()
    cbar.ax.set_aspect('auto')
    cbar.set_label('Meridional Geostrophic Velocity (m/s)')
    ax2 = cbar.ax.twiny()
    cbar.ax.set_position(pos)
    ax2.set_position(pos)
    cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_var,pad=0.1,extend='both')
    cbar2.ax.xaxis.set_ticks_position('top')
    cbar2.ax.set_xticklabels(['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$',
                          '$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    # cbar.ax.text(ssm-0.1*ssm,0.3,f'MIN\n{vari_min:.2f}',ha='center',va='center')
    # cbar.ax.text(mss+0.1*ssm,0.3,f'MAX\n{vari_max:.2f}',ha='center',va='center')
    ax.set_aspect('auto')
    text(0, 0, f'MIN = {vari_min:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
    text(1, 0, f'MAX = {vari_max:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
    if not (os.path.exists(ldir0 + f'vgos_{mon}_mean_global.png')):
        plt.savefig(ldir0 + f'vgos_{mon}_mean_global.png',bbox_inches='tight')
    figs.append([fig])
    plt.show()




# ##############################
# import matplotlib.pyplot as plt 
# import matplotlib.image as mgimg
# from matplotlib import animation

# ldir0 = '/media/hbatistuzzo/DATA/Ilhas/CMEMS/alt_cmems/processed/monthly_timeseries/'
# #### RIGHT, we need to take the mean of the means..
# fig = plt.figure()
# figs = []

# for mon in var.keys():
#     ## Read in picture
#     fname = f"adt_{mon}_mean_global.png"
#     print(f'now reading {fname}')
#     img = mgimg.imread(ldir0+fname)
#     imgplot = plt.imshow(img)
#     # append AxesImage object to the list
#     figs.append([imgplot])

# my_anim = animation.ArtistAnimation(fig, figs, interval=1000, blit=True, repeat_delay=1000)
# my_anim.save("animation.mp4")

# plt.show()


# #ugh lets try func animation then



### Rai method
import cv2
import argparse
import os
from pathlib import Path
paths = sorted(Path('/media/hbatistuzzo/DATA/Ilhas/Era5/wind/global/v10n/monthly/').iterdir(), key=os.path.getmtime)

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
args = vars(ap.parse_args())

# Arguments
dir_path = '/media/hbatistuzzo/DATA/Ilhas/Era5/wind/global/v10n/monthly/'
ext = args['extension']
output = args['output']

images = []
for f in sorted(os.listdir(dir_path),key=os.path.getctime):
    if f.endswith(ext):
        images.append(f)

# Determine the width and height from the first image
image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
cv2.imshow('video',frame)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 2.0, (width, height))

for image in images:

    image_path = os.path.join(dir_path, image)
    frame = cv2.imread(image_path)

    out.write(frame) # Write out frame to video

    cv2.imshow('video',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))


########################################################### Now for the STD
ldir0 = r"/media/hbatistuzzo/DATA/Ilhas/CMEMS/alt_cmems/processed/full_timeseries/"
path = 'full_means_stds.nc'
ds = xr.open_dataset(ldir0+path) # Means and STDs for ADT, SLA, UGOS, VGOS, UGOSA, VGOSA


# 2) Get some stats:
def stats(vari):
    mu = np.nanmean(vari.values)
    sigma = np.nanstd(vari.values)
    vari_min = np.nanmin(vari.values)
    vari_max = np.nanmax(vari.values)
    print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {vari_min:.2f}, max is {vari_max:.2f}')
    return mu, sigma, vari_min, vari_max

vari = ds.vgos_std
variv = ds.vgos_std.values # 720 x 1440 array float 32
mu,sigma,vari_min,vari_max=stats(vari)

# 3) Set colorbar intervals
[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

# 4) aux
lon = vari.lon.values # 1440 array float32
lat = vari.lat.values # 720 ~~~
ticks_var = [0,sm,m,ms,mss]
gfont = {'fontsize' : 20}

# 5) Wrap greenwich if necessary
variv, lon = shiftgrid(180., variv, lon, start=False) #0 to 360 needs wrapping


# 6) Plot
fig = plt.figure(figsize=(8,6),dpi=300)
ax = plt.axes(projection=ccrs.Robinson())

CS = plt.contour(lon, lat, variv, transform=ccrs.PlateCarree(),levels=[sm,m,ms,mss],
                  linewidths=0.25,colors='k',zorder=1,inline=True)
fmt = {} 
strs = ['$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']
for l, s in zip(CS.levels, strs): 
    fmt[l] = s
ax.clabel(CS, CS.levels, inline=True,fontsize=5,fmt=fmt)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1,zorder=2)
ax.add_feature(cfeature.LAND, zorder=3,edgecolor='black',linewidths=0.25)
ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1,zorder=4)
ax.add_feature(cfeature.BORDERS, linewidths=0.25,zorder=5)
ax.add_feature(cfeature.RIVERS, linewidths=0.25,zorder=6)
ax.set_global()
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
plt.title("CMEMS vgos 1993-2019 STD",fontdict = gfont,pad=10)
cf = plt.pcolormesh(lon,lat,variv,transform=ccrs.PlateCarree(),cmap=cmocean.cm.amp,
                    vmin=0,vmax = 0.5,zorder=0)
gl = ax.gridlines(draw_labels=True,xlocs=np.arange(-120, 121, 60),
                  ylocs=np.arange(-90, 91, 30),
                  linewidth=0.25,color='black',zorder = 7)
gl.xlabel_style = {'size': 10, 'color': 'black'}
gl.ylabel_style = {'size': 10, 'color': 'black'}
gl.rotate_labels = False
gl.ypadding = 30
gl.xpadding = 10
cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_var,pad=0.1,extend='max')
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_var,extend='max')
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.set_xticklabels(['0', '$\mu$-$\sigma$', '$\mu$',
                          '$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
lon_formatter = LongitudeFormatter(zero_direction_label=True)
# cbar.ax.text(ssm-abs((0.2*ssm)),0,f'MIN\n{vari_min:.2f}',ha='center',va='center')
# cbar.ax.text(mss+abs((0.2*ssm)),0,f'MAX\n{vari_max:.2f}',ha='center',va='center')
cbar.set_label('Meridional Geostrophic Velocity (m/s)')
cbar.ax.get_yaxis().set_ticks([])
# cbar.ax.text(ssm-0.1*ssm,0.3,f'MIN\n{vari_min:.2f}',ha='center',va='center')
# cbar.ax.text(mss+0.1*ssm,0.3,f'MAX\n{vari_max:.2f}',ha='center',va='center')
ax.set_aspect('auto')
text(1, 0, f'MAX = {vari_max:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
# if not (os.path.exists(ldir0 + '/global/vgos_full_mean_global.png')):
#     plt.savefig(ldir0 + '/global/vgos_full_mean_global.png',bbox_inches='tight')
plt.show()

#############################################################################################

#Monthly STD
ldir0 = r"/media/hbatistuzzo/DATA/Ilhas/CMEMS/alt_cmems/processed/monthly_timeseries/"
path = 'monthly_means_stds.nc'
ds = xr.open_dataset(ldir0+path) # Means and STDs for ADT, SLA, UGOS, VGOS, UGOSA, VGOSA

month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

adt_std = ds.adt_std_months
ugos_std = ds.ugos_std_months
vgos_std = ds.vgos_std_months

#plotting monthly means
lon = ds.lon.values
lat = ds.lat.values

#for vgos
z = vgos_std.values

vgos = {}
namespace = globals()
vgos_list=[]
for m in np.arange(0,12):
    vgos[month[m]] = vgos_std[m,:,:].values
    vgos_list.append(vgos_std[m,:,:].values) #this works for separating the months
    namespace[f'vgos_std_{month[m]}'] = vgos_std[m] #separates the 12 dataarrays by name
    print('this worked')


var = {}
namespace = globals()
var_list=[]
for m in np.arange(0,12):
    var[month[m]] = vgos_std[m,:,:].values
    var_list.append(vgos_std[m,:,:].values) #this works for separating the months
    namespace[f'vgos_mean_{month[m]}'] = vgos_std[m] #separates the 12 dataarrays by name
    print('this worked')

#for a fixed colorbar, lets get the mean of the means and the mean of the stds
mu = round(np.nanmean(z),4)
sigma = round(np.nanstd(z),4)
z_min = round(np.nanmin(z),4)
z_max = round(np.nanmax(z),4)
print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {z_min:.2f}, max is {z_max:.2f}')

[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])
# ticks_vgos = [ssm,sm,m,ms,mss]
ticks = [0,sm,m,ms,mss]
gfont = {'fontname':'Helvetica','fontsize' : 16}

for mon in vgos.keys():
    lon = ds.lon.values
    # 5) Wrap greenwich
    var[mon], lon = shiftgrid(180., var[mon], lon, start=False) #0 to 360 needs wrapping
    mu = round(np.nanmean(var[mon]),2)
    sigma = round(np.nanstd(var[mon]),2)
    vari_min = round(np.nanmin(var[mon]),2)
    vari_max = round(np.nanmax(var[mon]),2)
    print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {vari_min:.2f}, max is {vari_max:.2f}')
    [ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
              np.around(mu-sigma,decimals=2),
              np.around(mu,decimals=2),
              np.around(mu+sigma,decimals=2),
              np.around(mu+2*sigma,decimals=2)]
    print([ssm,sm,m,ms,mss])
    fig = plt.figure(figsize=(8,6),dpi=300)
    ax = plt.axes(projection=ccrs.Robinson())
    # z = vgos[mon] 
    # z, lons = shiftgrid(180., z, lons, start=False)
    ax = plt.axes(projection=ccrs.Robinson())
    # CS = plt.contour(lons, lats, z, transform=ccrs.PlateCarree(),linewidths=0.5,colors='k',
    #                  levels=[ssm,sm,m,ms,mss],zorder=1)
    # fmt = {} 
    # strs = ['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']
    # for l, s in zip(CS.levels, strs): 
    #     fmt[l] = s
    # ax.clabel(CS, CS.levels, inline=True,fontsize=5,fmt=fmt)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1,zorder=2)
    ax.add_feature(cfeature.LAND, zorder=3,edgecolor='black',linewidths=0.25)
    ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1,zorder=4)
    ax.add_feature(cfeature.BORDERS, linewidths=0.25,zorder=5)
    ax.add_feature(cfeature.RIVERS, linewidths=0.25,zorder=6)
    ax.set_global()
    plt.title(f'CMEMS vgos 1993-2019 {mon} STD',fontdict = {'fontsize' : 24},family='Verdana',pad=15);
    cf = plt.pcolormesh(lon,lat,var[mon],transform=ccrs.PlateCarree(),cmap=cmocean.cm.amp,
                    vmin=0,vmax=0.5,zorder=0)
    cbar = plt.colorbar(ax=ax,orientation="horizontal",extend='max',ticks=ticks,pad=0.1)
    cbar.set_label('Meridional Geostrophic Velocity (m/s)')
    pos = cbar.ax.get_position()
    cbar.ax.set_aspect('auto')
    ax2 = cbar.ax.twiny()
    cbar.ax.set_position(pos)
    ax2.set_position(pos)
    cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks,extend='max')
    cbar2.ax.xaxis.set_ticks_position('top')
    cbar2.ax.set_xticklabels(['0','$\mu$-$\sigma$','$\mu$','$\mu$+$\sigma$','2$\mu$+$\sigma$']) 
    # gl.xlabel_style = {'size': 15, 'color': 'gray'}
    # gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
    # gl.ylocator = mticker.FixedLocator([-40,-30,-20,-10,0])
    # lon_formatter = LongitudeFormatter(zero_direction_label=True)
    gl = ax.gridlines(draw_labels=True,xlocs=np.arange(-120, 121, 60),
                  ylocs=np.arange(-90, 91, 30),
                  linewidth=0.25,color='black',zorder = 7)
    # gl.xlocator = mticker.FixedLocator([-180, -90,-45, 0, 45, 90,180])
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}
    gl.rotate_labels = False
    gl.ypadding = 30
    gl.xpadding = 10
    cbar.ax.get_yaxis().set_ticks([])
    # cbar.ax.text(ssm-2,0.3,f'MIN\n{z_min:.2f}',ha='center',va='center')
    # cbar.ax.text(mss+2,0.3,f'MAX\n{z_max:.2f}',ha='center',va='center')
    # ax.set_aspect('auto')
    ax.set_aspect('auto')
    # text(0, 0, f'MIN = {z_min:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
    text(1, 0, f'MAX = {z_max:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
    if not (os.path.exists(ldir0 + f'vgos_{mon}_std_global.png')):
        plt.savefig(ldir0 + f'vgos_{mon}_std_global.png',bbox_inches='tight')
    plt.show()
plt.show()

##### movie time
import cv2
import argparse
import os
from pathlib import Path
paths = sorted(Path('/media/hbatistuzzo/DATA/Ilhas/CMEMS/alt_cmems/processed/monthly_timeseries/global/vgos/std/').iterdir(), key=os.path.getmtime)

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
args = vars(ap.parse_args())

# Arguments
dir_path = '/media/hbatistuzzo/DATA/Ilhas/CMEMS/alt_cmems/processed/monthly_timeseries/global/vgos/std/'
ext = args['extension']
output = args['output']

images = []
for f in sorted(os.listdir(dir_path),key=os.path.getctime):
    if f.endswith(ext):
        images.append(f)

# Determine the width and height from the first image
image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
cv2.imshow('video',frame)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 2.0, (width, height))

for image in images:

    image_path = os.path.join(dir_path, image)
    frame = cv2.imread(image_path)

    out.write(frame) # Write out frame to video

    cv2.imshow('video',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))





#################################################################################
# Ilhas 

# 1) CMEMS FULL SERIES PLOTS
ldir0 = r"/media/hbatistuzzo/DATA/Ilhas/CMEMS/alt_cmems/processed/full_timeseries/"
path = 'full_means_stds.nc'
ds = xr.open_dataset(ldir0+path) # Means and STDs for ADT, SLA, UGOS, VGOS, UGOSA, VGOSA

#Define the region
latN = 10.125
latS = -10.125
lonW = 295.125
lonE = 15.375

vari = ds.vgosa_mean
lon = vari.lon.values # 1440 array float32
lat = vari.lat.values # 720 ~~~

vari_mean1 = vari.sel(lat=slice(latS,latN), lon=slice(lonW,359.75))
vari_mean2 = vari.sel(lat=slice(latS,latN), lon=slice(0,lonE))
vari_mean_ilhas = xr.concat([vari_mean1, vari_mean2], dim='lon')
vari = vari_mean_ilhas.values #new vari!!!!

#### the new lon needs to be this size!
lon = vari_mean_ilhas.lon.values 
lat = vari_mean_ilhas.lat.values 

# 2) Get some stats:
mu, sigma, vari_min, vari_max = stats(vari)

# 3) Set colorbar intervals
[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

# 4) aux
ticks_var = [ssm,sm,m,ms,mss]
gfont = {'fontsize' : 20}
# 'Absolute Dynamic Topography (m)','Zonal Geostrophic Velocity (m/s)', 'Meridional Geostrophic Velocity (m/s)'

fig = plt.figure(figsize=(8,6),dpi=300)
ax = plt.axes(projection=ccrs.PlateCarree())
arch = ax.plot(-29.34583333,0.91694444, 'k*', markersize=3, markeredgewidth = 0.25, transform=ccrs.PlateCarree()) # WHY
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1)
ax.add_feature(cfeature.LAND, zorder=1,edgecolor='black',linewidths=0.25)
ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1)
ax.add_feature(cfeature.BORDERS, linewidths=0.25)
ax.add_feature(cfeature.RIVERS, linewidths=0.5)
ax.set_extent([-65, 15, -10.1, 10.1], crs=ccrs.PlateCarree())
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
plt.title("CMEMS VGOSA 1979-2020 mean",fontdict = gfont)
# CS = plt.contour(lon, lat, vari, transform=ccrs.PlateCarree(),levels=[sm,m,ms],
#                   linewidths=0.5,colors='k',zorder=1,inline=True)
# fmt = {} 
# # strs = ['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']
# strs = ['$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$']
# for l, s in zip(CS.levels, strs): 
#     fmt[l] = s
# ax.clabel(CS, CS.levels, inline=True,fontsize=5,fmt=fmt)
cf = plt.pcolormesh(lon,lat,vari,transform=ccrs.PlateCarree(),cmap=cmocean.cm.balance,
                    vmin=ssm,vmax=mss,zorder=0)
cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_var,extend='both',pad=0.1,shrink=0.9)
cbar.set_label('Meridional Geostrophic Velocity Anomaly (m/s)')
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_var,extend='both',pad=0.1,shrink=0.9)
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.set_xticklabels(['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$',
                          '$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=0.25, color='black', alpha=0.5, linestyle='dashed')
gl.xlabels_top = False
gl.ylabels_right = False
# gl.xlines = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-10,-5,0,5,10])
cbar.ax.get_yaxis().set_ticks([])
text(0, -0.8, f'MIN = {vari_min:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
text(1, -0.8, f'MAX = {vari_max:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
# if not (os.path.exists(ldir1 + 'u10n_full_mean_ilhas.png')):
#     plt.savefig(ldir1 + 'u10n_full_mean_ilhas.png',bbox_inches='tight')
plt.show()


############################### now the monthly ilhas stuff
ldir0=r"/media/hbatistuzzo/DATA/Ilhas/CMEMS/alt_cmems/processed/monthly_timeseries/"
ds = xr.open_dataset(ldir0+'monthly_means_stds.nc')

month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

#Define the region
latN = 10.125
latS = -10.125
lonW = 295.125
lonE = 15.375

vari = ds.vgos_mean_months #12 x 720 x 1440
lon = vari.lon.values # 1440 array float32
lat = vari.lat.values # 720 ~~~

vari_mean1 = vari.sel(lat=slice(latS,latN), lon=slice(lonW,359.75))
vari_mean2 = vari.sel(lat=slice(latS,latN), lon=slice(0,lonE))
vari_mean_ilhas = xr.concat([vari_mean1, vari_mean2], dim='lon')
vari = vari_mean_ilhas.values #new vari!!!!

#### the new lon needs to be this size!
lon = vari_mean_ilhas.lon.values 
lat = vari_mean_ilhas.lat.values 

#Get some stats:
mu, sigma, vari_min, vari_max = stats(vari)

# 3) Set FIXED colorbar intervals
[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

# 4) aux
ticks_var = [ssm,sm,m,ms,mss]
gfont = {'fontsize' : 20}
# 'Absolute Dynamic Topography (m)','Zonal Geostrophic Velocity (m/s)', 'Meridional Geostrophic Velocity (m/s)'


var = {}
namespace = globals()
var_list=[]
for m in np.arange(0,12):
    var[month[m]] = vari_mean_ilhas[m,:,:].values
    var_list.append(vari_mean_ilhas[m,:,:].values) #this works for separating the months
    namespace[f'vgos_mean_{month[m]}'] = vari_mean_ilhas[m] #separates the 12 dataarrays by name
    print('this worked')


figs = []
for mon in var.keys():
    fig = plt.figure(figsize=(8,6),dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())
    arch = ax.plot(-29.34583333,0.91694444, 'k*', markersize=3, markeredgewidth = 0.25, transform=ccrs.PlateCarree()) # WHY
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1)
    ax.add_feature(cfeature.LAND, zorder=1,edgecolor='black',linewidths=0.25)
    ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1)
    ax.add_feature(cfeature.BORDERS, linewidths=0.25)
    ax.add_feature(cfeature.RIVERS, linewidths=0.5)
    ax.set_extent([-65, 15, -10.1, 10.1], crs=ccrs.PlateCarree())
    states_provinces = cfeature.NaturalEarthFeature(category='cultural',
                                                    name='admin_1_states_provinces_lines',
    scale='50m',facecolor='none')
    ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
    plt.title(f"CMEMS VGOS 1993-2019 {mon} mean",fontdict = gfont)
    cf = plt.pcolormesh(lon,lat,var[mon],transform=ccrs.PlateCarree(),cmap=cmocean.cm.balance,
                    vmin=ssm,vmax=mss,zorder=0)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=0.25, color='black', alpha=0.5, linestyle='dashed')
    gl.xlabels_top = False
    gl.ylabels_right = False
    # gl.xlines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylocator = mticker.FixedLocator([-10,-5,0,5,10])
    cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_var,pad=0.1,extend='both')
    pos = cbar.ax.get_position()
    cbar.ax.set_aspect('auto')
    cbar.set_label('Meridional Geostrophic Velocity (m/s)')
    ax2 = cbar.ax.twiny()
    cbar.ax.set_position(pos)
    ax2.set_position(pos)
    cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_var,pad=0.1,extend='both')
    cbar2.ax.xaxis.set_ticks_position('top')
    cbar2.ax.set_xticklabels(['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$',
                          '$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    text(0, -0.75, f'MIN = {vari_min:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
    text(1, -0.75, f'MAX = {vari_max:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
    if not (os.path.exists(ldir0 + f'vgos_{mon}_mean_ilhas.png')):
        plt.savefig(ldir0 + f'vgos_{mon}_mean_ilhas.png',bbox_inches='tight')
    figs.append([fig])
    plt.show()


##### movie time
import cv2
import argparse
import os
from pathlib import Path
paths = sorted(Path('/media/hbatistuzzo/DATA/Ilhas/CMEMS/alt_cmems/processed/monthly_timeseries/ilhas/ilhas_mean/').iterdir(), key=os.path.getmtime)

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
args = vars(ap.parse_args())

# Arguments
dir_path = '/media/hbatistuzzo/DATA/Ilhas/CMEMS/alt_cmems/processed/monthly_timeseries/ilhas/ilhas_mean/'
ext = args['extension']
output = args['output']

images = []
for f in sorted(os.listdir(dir_path),key=os.path.getmtime):
    if f.endswith(ext):
        images.append(f)

# Determine the width and height from the first image
image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
cv2.imshow('video',frame)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 2.0, (width, height))

for image in images:

    image_path = os.path.join(dir_path, image)
    frame = cv2.imread(image_path)

    out.write(frame) # Write out frame to video

    cv2.imshow('video',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))







