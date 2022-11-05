#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 09:34:31 2020

@author: polito
"""
from ftplib import FTP
import os
import glob
import sys
import logging

# =============================================================================
logging.basicConfig(filename=__file__.replace('.py', '.log'),
                    format="%(asctime)-15s %(levelname)-8s %(message)s",
                    level=logging.INFO, filemode='a+')
# =============================================================================


def prilog(str):
    print(str)
    logging.info(str)
    return


# =============================================================================
# local output directory path
ldir0 = "/data2/cmems_alt/"
# CMEMS hostname
host = "my.cmems-du.eu"
# CMEMS product numbers. Detailed description is in their web site
product = "008_047"
# remote directory path, user and password
dir0 = "/Core/SEALEVEL_GLO_PHY_L4_REP_OBSERVATIONS_" + \
        product + "/dataset-duacs-rep-global-merged-allsat-phy-l4/"
user = "hbatistuzzo"
pawd = "Ro$$by88"
# open connection and move to the directory that has years as subdirectories
ftp0 = FTP(host=host, user=user, passwd=pawd)
ftp0.cwd(dir0)

# get list of existing directories, this is a list of strings like ["1993", "1994", ...]
ys = ftp0.nlst() #returns a list of file names in specified directory

for y in ys:
    prilog('Doing year {:}.'.format(y))
    ldir1 = ldir0 + y #eg "/data2/cmems_alt/1993"
    # is there a local directory for this year? if not, happy new year!
    if not(os.path.exists(ldir1)):
        os.mkdir(ldir1)
    
    # move to the next directory/year
    ftp0.cwd(y)

    #### new loop for months ####
    ms = ["%.2d" % i for i in range(1,13)] #returns a list of months (01 to 12)
    for m in ms:
        prilog('Doing month {:}.'.format(m))
        ldir2 = ldir1 + "/" + m #eg "/data2/cmems_alt/1993/01"
        # is there a directory for months? if not, happy new month!
        if not(os.path.exists(ldir2)):
            os.mkdir(ldir2)
            # move to next directory/month
        ftp0.cwd(m)
        # get a list of files
        fs = ftp0.nlst()
        # filenames should be like 'dt_global_allsat_phy_l4_20121208_20190101.nc
        for f in fs:
            # check if the files on the ftp site follow the usual convention
            if f[0:28] != 'dt_global_allsat_phy_l4_' + y:
                prilog('Error: Check {:} Name convention was changed?'
                       .format(f))
                sys.exit(1)
                # check the file size (should be near 7MB)
                fsize = ftp0.size(f)
            if fsize < 7000000:
                prilog('Warning: File {:} is too small, download interrupted?'
                       .format(f))
                
                # Check if the file exists, and if it is outdated
                # s is a regex in that the * covers the creation date in the local file
            s = ldir2 + '/' + f[:33] + '*.nc'
            # locf is the (old) local file
            locf = glob.glob(s)
            lelo = len(locf)
            # outf is the output file and equals the one in the remote site
            outf = ldir2 + '/' + f
            if lelo == 0:
                # local file does not exist, probably a new year, download it
                ftp0.retrbinary("RETR " + f, open(outf, 'wb').write)
                prilog('0 - Downloaded {:}'.format(outf))
                # if there are leftover files from previous versions, stop
            elif lelo > 1:
                prilog('Error: You have {:} files named {:}, will remove all.'
                       .format(lelo, s))
                for garb in locf:
                    os.remove(garb)
                ftp0.retrbinary("RETR " + f, open(outf, 'wb').write)
                prilog('1 - Downloaded a single  {:}'.format(outf))
            elif locf[0] != outf:
                # file for that day exists, but is outdated and will be removed
                os.remove(locf[0])
                # and a fresh one can be downloaded
                ftp0.retrbinary("RETR " + f, open(outf, 'wb').write)
                prilog('2 - Downloaded a newer {:}'.format(outf))
                # if names match, but sizes don't, remove and download again
            elif ((locf[0] == outf) &
                  (os.stat(locf[0]).st_size != fsize)):
                # local file too small, download again
                os.remove(locf[0])
                # and a fresh one can be downloaded
                ftp0.retrbinary("RETR " + f, open(outf, 'wb').write)
                prilog('3 - Sizes differ, downloaded {:}'.format(outf))
            else:
                prilog('4 - File {:} was already there.'.format(outf))
                    
        # Down one level in the directory tree
        ftp0.cwd('..') #back to directory of months
    

    # Down one level in the directory tree
    ftp0.cwd('..')

ftp0.quit()
prilog('Done.')
