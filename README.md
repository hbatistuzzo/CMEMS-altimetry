# CMEMS_altimetry
 

![GitHub top language](https://img.shields.io/github/languages/top/hbatistuzzo/CMEMS_altimetry)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/hbatistuzzo/CMEMS_altimetry)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/hbatistuzzo/CMEMS_altimetry)
![GitHub last commit](https://img.shields.io/github/last-commit/hbatistuzzo/CMEMS_altimetry)

- Python 3.8.13
	- Numpy 1.20.3
	- Seaborn 0.11.2
	- Matplotlib 3.5.3
	- Xarray 2022.10.0
	- Dask 2021.09
	- Cartopy 0.21.0
	- Cmocean 2.0

<img src="cmems.jpg" align="right" width="50%"/>
The Copernicus Marine Environment Monitoring Service (CMEMS), implemented and operated by Mercator Ocean, provides oceanographic products and services for maritime safety, coastal and marine environment, climate and weather forecasting and marine resources users.

<br/>
In this project I've created some functions to help plot global monthly climatologies of zonal and meridional geostrophic velocities. The absolute dynamic topography variable was used to construct hovmöller diagrams in latitudes with strong signals in the low-frequency energy spectrum. Rossby waves can then be visualized in this format
<p align="center"><img src="vgos.gif"alt="full"  width="75%"></p>


There is also some snippets to plot "zoomed-in" data in a regular lat-long cartesian projection. Even before the identification of planetary waves by FIR-2D filtering, the monthly climatology of meridional geostrophic velocities already betrays their existence as chains of crests and throughs (e.g. around 5ºN):

<p align = "center">
<img src="vgos_Jan_mean_ilhas.png" alt="ilhas3" width="75%">
</p>

<p align = "center">
The star marks the location of the St. Peter & St. Paul archipelago
</p>

<br/>
It uses dask to optimize the code run (as opening these many netcdf files in nested concatenation is not a trivial task). Even less trivial is performing the chronological
mean for the monthly climatology. Big kudos to those who created this awesome tool.
<br/>
Finally, it also spits out Hovmoller diagrams for each of the latitudes of interest, since they are useful for identifying meso and large-scale phenomena such as vortices and planetary (Rossby/Equatorial Kelvin) waves.
<br/>
I managed to do some neat animations with matplotlib and cartopy!
<p align="center"><img src="https://im3.ezgif.com/tmp/ezgif-3-058b56cfa6.gif" width="100%" alt="cake"></p>