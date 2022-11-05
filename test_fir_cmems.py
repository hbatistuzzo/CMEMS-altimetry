#!/home/paulo/miniconda3/bin/python3.7
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 09:00:04 2020

@author: paulo
"""

import numpy as np
from fir import maxvar, varexp, f_longterm, f_annual, get_zg, f_wave, get_cLTA, get_aux,\
     get_Rrad, get_basin, get_gparms
from collections import OrderedDict as od
from xarray import Dataset
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.dates import YearLocator, DateFormatter
from matplotlib.ticker import StrMethodFormatter
# -----------------------------------------------------------------------------


def save_nc():
    '''
    Saves the output file, variables are specific to this program
    '''
    outfile = '/data0/cmems/conv_filt/xr_hovs_conv_filt_{:}_{:06.3f}.nc'.format(bas, lat)
    print('[save_nc] Saving {:}'.format(outfile))
    ddout = {}  # clean ddout just in case
    ddout = {'time': {'dims': 'time', 'data': jtime, 'attrs': {'units': 'd'}},
             'lat': {'dims': [], 'data': lat, 'attrs': {'units': 'deg N/S'}},
             'lon': {'dims': 'lon', 'data': xio, 'attrs': {'units': 'deg E/W'}},
             'bas': {'dims': [], 'data': bas, 'attrs': {'descr': 'basin name'}}}
    encoding = {}
    ddout.update({'Rd': {'dims': [], 'data': Rrio,
                         'attrs': {'units': 'km', 'descr': 'mean 1st Rossby radius, OSU'}}})
    encoding.update({'Rd': {'dtype': 'int16', 'scale_factor': 0.1,
                            'zlib': True, '_FillValue': -9999}})
    ddout.update({'c1': {'dims': [], 'data': c1io,
                         'attrs': {'units': 'km/d', 'descr': 'mean 1st g-wave cp, OSU'}}})
    encoding.update({'c1': {'dtype': 'int16', 'scale_factor': 0.1,
                            'zlib': True, '_FillValue': -9999}})
    ddout.update({'cp_lin': {'dims': [], 'data': cp_lin, 'attrs': {'units': 'km/d',
                             'descr': 'mean 1st R-wave cp, linear'}}})
    encoding.update({'cp_lin': {'dtype': 'int16', 'scale_factor': 0.01,
                                'zlib': True, '_FillValue': -9999}})
    ddout.update({'wc': {'dims': [], 'data': wcrit, 'attrs': {'units': '1/d',
                         'descr': 'mean 1st mode R-wave critical frequency'}}})
    encoding.update({'wc': {'dtype': 'int16', 'scale_factor': 0.01,
                            'zlib': True, '_FillValue': -9999}})
    for key in z_c.keys():
        ddout.update({'z_'+key: {'dims': ('time', 'lon'), 'data': z_c[key],
                                 'attrs': {'units': 'mm'}}})
        encoding.update({'z_'+key: {'dtype': 'int16', 'scale_factor': 0.01, 'zlib': True,
                                    '_FillValue': -9999}})
        ddout.update({'A_'+key: {'dims': [], 'data': A[key],
                                 'attrs': {'units': 'mm'}}})
        encoding.update({'A_'+key: {'dtype': 'int16', 'scale_factor': 0.01,
                                    'zlib': True, '_FillValue': -9999}})
        ddout.update({'vx_'+key: {'dims': [], 'data': vx[key],
                                  'attrs': {'units': 'unitless'}}})
        encoding.update({'vx_'+key: {'dtype': 'int16', 'scale_factor': 0.01,
                                     'zlib': True, '_FillValue': -9999}})
    for key in L.keys():
        ddout.update({'L_'+key: {'dims': [], 'data': L[key], 'attrs': {'units': 'km'}}})
        encoding.update({'L_'+key: {'dtype': 'int16', 'scale_factor': 1.0,
                                    'zlib': True, '_FillValue': -9999}})
        ddout.update({'Le_'+key: {'dims': [], 'data': Le[key], 'attrs': {'units': 'km'}}})
        encoding.update({'Le_'+key: {'dtype': 'int16', 'scale_factor': 1.0,
                                     'zlib': True, '_FillValue': -9999}})
        ddout.update({'T_'+key: {'dims': [], 'data': T[key], 'attrs': {'units': 'km'}}})
        encoding.update({'T_'+key: {'dtype': 'int16', 'scale_factor': 0.01,
                                    'zlib': True, '_FillValue': -9999}})
        ddout.update({'Te_'+key: {'dims': [], 'data': Te[key], 'attrs': {'units': 'd'}}})
        encoding.update({'Te_'+key: {'dtype': 'int16', 'scale_factor': 0.01,
                                     'zlib': True, '_FillValue': -9999}})
        ddout.update({'cp_'+key: {'dims': [], 'data': cp[key], 'attrs': {'units': 'km'}}})
        encoding.update({'cp_'+key: {'dtype': 'int16', 'scale_factor': 0.01,
                                     'zlib': True, '_FillValue': -9999}})
        ddout.update({'cpe_'+key: {'dims': [], 'data': cpe[key], 'attrs': {'units': 'km'}}})
        encoding.update({'cpe_'+key: {'dtype': 'int16', 'scale_factor': 0.01,
                                      'zlib': True, '_FillValue': -9999}})
    ds = Dataset.from_dict(ddout)
    ds.to_netcdf(outfile, format='NETCDF4', encoding=encoding)
# -----------------------------------------------------------------------------


def write_report():
    '''
    Write a little report, variables are specific to this program
    '''
    for key in z_c.keys():
        print('------------------------ write_report -----------------------------')
        if key[0:2] == 'R_':
            print('{:} -> cp=({:5.2f} +- {:5.2f})km/d,'.format(key, cp[key], cpe[key]))
            print('L=({:7.1f} +- {:7.1f})km,'.format(L[key], Le[key]))
            print('T=({:5.1f} +- {:5.1f})d,'.format(T[key], Te[key]))
            print('A={:5.1f}mm'.format(A[key]))
        else:
            print('{:} -> A={:5.1f}mm'.format(key, A[key]))
        print('vx={:5.1f}%'.format(vx[key]))
        print('--------------------------------------------------------------------')
# -----------------------------------------------------------------------------


def plot_hovs():
    '''
    Plots the Hovmollers, variables are specific to this program
    '''
    rc('image', cmap='RdBu_r')
    fig, ax0 = plt.subplots(1, len(z_c), sharey=True,
                            figsize=(16, 10), dpi=100)
    aa = np.array([A[key] for key in z_c.keys()])[3:-1].max()

    for a, key in enumerate(z_c.keys()):
        zm = z_c[key].mean()
        zs = z_c[key].std()
        if a <= 1:
            vmin = zm - 2*zs
            vmax = zm + 2*zs
        elif (a < len([key for key in z_c.keys()])-1):
            vmin = -aa
            vmax = aa
        else:
            vmin = -2*zs
            vmax = 2*zs
        pc = ax0[a].pcolormesh(xio, time, z_c[key], vmin=vmin, vmax=vmax)
        plt.colorbar(pc, ax=ax0[a], orientation='horizontal',
                     extend='both', pad=0.03, fraction=0.03, format='%d',
                     ticks=np.linspace(vmin, vmax, 5))
# format the ticks etc.
        ax0[a].set_title(r'{:} $\sigma^2_p$={:3.1f}'.format(
                key.capitalize(), vx[key]))
        ax0[a].xaxis.set_major_locator(plt.MultipleLocator(
                int((xio[-1]-xio[0])/30)*10))
        ax0[a].xaxis.set_minor_locator(plt.MultipleLocator(
                int((xio[-1]-xio[0])/30)*5))
        if a == 0:
            iyea = YearLocator()
            years_fmt = DateFormatter('%Y')
        ax0[a].xaxis.set_major_formatter(StrMethodFormatter(u"{x:.0f}°W"))
        ax0[a].yaxis.set_major_locator(iyea)
        ax0[a].yaxis.set_major_formatter(years_fmt)
        ax0[a].yaxis.grid(alpha=0.5)
        ax0[a].xaxis.grid(alpha=0.5)
    plt.tight_layout(w_pad=0, h_pad=1)
    return fig, ax0
# -----------------------------------------------------------------------------


def select_basin(bas, y, lat, beta):
    # mask number of the basin (0 -> continent)
    io = 1 + np.where(np.array(basin) == bas)[0][0]
    nf = np.where(y == lat)[0][0]
    # slice according to latitude index
    xb = mask[nf, :]
    Rrb = Rrg[nf, :]
    c1b = c1g[nf, :]
    # find indices of xb that correspond to this basin (io)
    ib = np.where(xb == io)[0]
    # crop longitudes, Rossby radii and gravity wave phase speeds
    # to current basin
    xio = x[ib]
    Rrio = Rrb[ib].mean()
    c1io = c1b[ib].mean()
    # get 1st mode long Rossby wave parameters
    cp_lin = -beta*Rrio**2
    # critical frequency waves
    wcrit = np.sqrt(np.abs(cp_lin)*Omega/(2*Re))
    # set minimum period for filtering as half of the critical period or 20
    # days (~two TOPEX cycles = Nyquist period).
    T_lim = np.max([1/wcrit, 20])

    # check if there is enough data to be processed
    nxio = len(xio)
    if nxio <= minxlen:
        print('[select_basin] 1 - Narrow basin ({:} on the {:}), nothing to do.'.format(lat,
              bas.capitalize()))
        return ([-9999], 0, 0, 0, 0, 0, 0)
    # If necessary, wrap the map (Atlantic)
    if np.diff(ib).max() > 1.0:
        # find the index of the longitude closest to Greenwich
        gwi = np.where(np.diff(ib) == np.diff(ib).max())[0][0]
        # cut and paste the indices to get increasing longitudes
        ib = np.concatenate([ib[gwi+1:], ib[0:gwi+1]])
        xio = x[ib]
        xio[xio > 180.] = xio[xio > 180.]-360.
    # crop SSHA to the current basin
    # and set masked z values to z.fill_value
    z = zg[:, ib].filled()
    # fix mismatch between continental mask and cmems data
    zm0 = z.mean(axis=0)
    mm0 = np.where(zm0 != zg.fill_value)[0]
    # check if enough data remains to be processed
    if len(mm0) <= minxlen:
        print('[select_basin] 2 - Narrow basin ({:} on the {:}), nothing to do.'.format(lat,
              bas.capitalize()))
        return ([-99999], 0, 0, 0, 0, 0, 0)
    # crop the empty edges
    i0 = np.min(mm0)
    i1 = 1 + np.max(np.where(zm0 != zg.fill_value))
    if (i0 != 0) or (i1 != nxio):
        z = z[:, i0:i1]
        xio = xio[i0:i1]
        nxio = len(xio)
    # check if enough data remains to be processed
    if nxio <= minxlen:
        print('[select_basin] 3 - Narrow basin ({:} on the {:}), nothing to do.'.format(
              lat, bas.capitalize()))
        return ([-99999], 0, 0, 0, 0, 0, 0)
    return (z, xio, Rrio, c1io, cp_lin, wcrit, T_lim)
# -----------------------------------------------------------------------------


def filter_z(z, T_lim, cp_lin):
    z_c, L, Le, T, Te, cp, cpe, A, vx = od(), od(), od(), od(), od(), od(), od(), od(), od()
    z_c['ori'] = z.copy()
    A['ori'] = np.sqrt(2)*np.std(z_c['ori'])

    z_c['longterm'] = f_longterm(z, ny=5, delt=delt)
    z_c['longterm'] = maxvar(z_c['longterm'], z)*z_c['longterm']
    A['longterm'] = np.sqrt(2)*np.std(z_c['longterm'])
    z = z_c['ori'] - z_c['longterm']
    print('[filter_z] Long term signal extracted.')

    z_c['annual'] = f_annual(z, delt)
    z_c['annual'] = maxvar(z_c['annual'], z)*z_c['annual']
    A['annual'] = np.sqrt(2)*np.std(z_c['annual'])
    z = z - z_c['annual']
    print('[filter_z] Annual signal extracted.')

    # periods to filter for Rossby wave components
    Ps = np.round(365.25*np.array([2, 1, 0.5, 0.25, 0.125, 0.0625]))
    # exclude supercritical waves (perios < 2* critical period)
    Ps = Ps[(Ps > T_lim)]
    for T0 in Ps:
        Rc = 'R_' + str(int(T0))
        L0 = cp_lin*T0
        z_c[Rc] = f_wave(z, delx, delt, lat, L0, T0)
        z_c[Rc] = maxvar(z_c[Rc], z)*z_c[Rc]
        cp[Rc], cpe[Rc], L[Rc], Le[Rc], T[Rc], Te[Rc], A[Rc] = \
            get_cLTA(z_c[Rc], delx, delt, lat)
        print('[filter_z] Rossby wave band of {:}d extracted.'.format(T0))
        cp[Rc] = np.sign(L0)*cp[Rc]
        L[Rc] = np.sign(L0)*L[Rc]
        z = z - z_c[Rc]

    z_c['cr'] = f_wave(z, delx, delt, lat, Rrio, T_lim)
    z_c['cr'] = maxvar(z_c['cr'], z)*z_c['cr']
    cp['cr'], cpe['cr'], L['cr'], Le['cr'], T['cr'], Te['cr'], A['cr'] = \
        get_cLTA(z_c['cr'], delx, delt, lat)
    print('[filter_z] Critical freq. Rossby wave band extracted.')
    cp['cr'] = np.sign(L0)*cp['cr']
    L['cr'] = np.sign(L0)*L['cr']
    z = z - z_c['cr']

    z_c['residual'] = z.copy()
    A['residual'] = np.sqrt(2)*np.std(z_c['residual'])
    for key in z_c.keys():
        vx[key] = 100*varexp(z_c['ori'], z_c[key])

    return z_c, L, Le, T, Te, cp, cpe, A, vx
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# latitudes to be processed
lats = [-35.125, ]
# basins to be processed
basins = ['atlantic', ]
# minimum number of points in the longitudinal direction worth processing
minxlen = 25

delt = 1     # temporal grid spacing in days
delx = 0.25  # longitudinal grid spacng in degrees
# -----------------------------------------------------------------------------
# read coordinates and shapes
nx, ny, nt, x, y, time, jtime = get_aux()
# load global Rossby radii and c obtained (OSU) from climatology
Rrg, c1g = get_Rrad()
# load basin mask
mask, basin = get_basin()
# preallocate ordered dictionaries for the components and their corresponding parameters

# big loops
for lat in lats:
    # Parameters
    f0, beta, Omega, Re = get_gparms(lat)
    for bas in basins:
        # load global data 2D array to be filtered
        zg = get_zg(lat)
        print('[main] Processing lat={:} of the {:}'.format(lat, bas.capitalize()))
        (z, xio, Rrio, c1io, cp_lin, wcrit, T_lim) = select_basin(bas, y, lat, beta)
        if len(z) == 1:
            continue
        z_c, L, Le, T, Te, cp, cpe, A, vx = filter_z(z, T_lim, cp_lin)

        # print the parameters
        write_report()

        # Saving
        save_nc()

        # Plotting
        fig, ax0 = plot_hovs()
