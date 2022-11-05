#!/home/polito/miniconda2/envs/filtw/bin/python3
# -*- coding: utf-8 -*-

# ==================================================================
# This program runs after xy2xt_xr_make_annual_hovmollers.py
# It reads all annual
# Hovmoller diagrams of h(nt,nx) and glues all together in big
# Hovmoller diagrams that span 25 years. These are the basic files
# to run filters, display, etc.
# ==================================================================

import numpy as np
import netCDF4 as nc
import xarray as xr
import multiprocessing as mp
import os
import time
print("imported packages...")


def gluethem(lat):
    # input directories
    rd = r"/data1/cmems_alt/[1-2][0-9][0-9][0-9]/hovmollers"
    # output dir and files
    od = r"/data1/cmems_alt/big_hovmollers/xr_global_allsat_phy_l4_"
    rf = rd + '/xr_global_allsat_phy_l4_[1-2][0-9][0-9][0-9]_{0:07.3f}'\
        .format(lat) + '.nc'
    fl = nc.glob(rf)
    fl.sort()
    sy0 = fl[0][17:21]
    sy1 = fl[-1][17:21]
    wrf = od + '{0:07.3f}_{1:}_{2:}'.format(lat, sy0, sy1) + '.nc'
    pid = os.getpid()
    # read each altimeter Hovmoller data
    print("[{:}] -- Start --------------------".format(pid))
    print("[{:}] Reading {:} files for lat = {:}".format(pid, len(fl), lat))
    ds = xr.open_mfdataset(fl, combine='nested', concat_dim="time")
    print("[{:}] Writing {:}".format(pid, wrf))
    ds.to_netcdf(path=wrf, mode='w', format='NETCDF4')
    print("[{:}] -- End ----------------------".format(pid))
    ds.close()


if __name__ == '__main__':
    jobs = []

    lats = np.arange(-66.125, 66.125+0.25, 0.25)
    ncores = mp.cpu_count()

    # set up the reading file names (fl)
    for lat in lats:
        # for lat in lats[300:301]:
        # assemble file names
        p = mp.Process(target=gluethem, args=(lat,))
        jobs.append(p)
        p.start()
        # start one process per minute to let the load average increase
        time.sleep(10)
        # Do not start another year if system load is high
        while os.getloadavg()[0] > 50.0:
            print('Load average is > 50.0, sleeping 10 more seconds...')
            time.sleep(10)

# hack to fix incomplete files
# lats = [001.125, 002.875, 003.125, 003.625, 003.875 ....]
