bash wget-20210401081342.sh -H esgdata.gfdl.noaa.gov
https://esgf-node.llnl.gov/esgf-idp/openid/wujingyi38@gmail.com
wujingyi38@gmail.com
Qwer12345!

bash ./CMIP_wgets/wget-NorESM1-M.sh -s -o https://esgf-node.llnl.gov/esgf-idp/openid/wujingyi38@gmail.com

bash ../CMIP_wgets/wget-GFDL-ESM2G.sh -s -o https://esgf-node.llnl.gov/esgf-idp/openid/wujingyi38@gmail.com

bash ../CMIP_wgets/wget-CNRM-CM5.sh -s -o https://esgf-node.llnl.gov/esgf-idp/openid/wujingyi38@gmail.com

# NCO

> ncks -d lon,minimum_lon,maximum_lon in.nc out.nc

ncks -d lon,0,0 tos_Omon_HadCM3_historical_r4i1p1_185912-188411.nc test0.nc

[0,360] flatten dataset
CanCM4
- HadGEM2-AO
- HadGEM2-ES
- HadGEM2-CC
- HadCM3


## NOAA
wget ftp://ftp.cdc.noaa.gov/Datasets/noaa.oisst.v2/lsmask.nc
wget ftp://ftp.cdc.noaa.gov/Datasets/noaa.oisst.v2/sst.wkmean.1990-present.nc


https://www.psl.noaa.gov/cgi-bin/db_search/DBSearch.pl?Dataset=NCEP+GODAS&Variable=potential+temperature&group=0&submit=Search

wget 'http://apdrc.soest.hawaii.edu/erddap/griddap/hawaii_soest_c71f_e12b_37f8.nc?temp[(1871-01-15):1:(1871-12-15)][(20):1:(20)][(-75.25):1:(89.25)][(0.25):1:(359.75)]'


> ncks -d time,start_time,end_time in.nc out.nc

ncks -d time,"1861-01-01","1963-12-30" tos_Omon_HadGEM2-AO_historical_r1i1p1_186001-200512.nc ./tos_Omon_HadGEM2-AO_historical_r1i1p1_186101-196312_full.nc


ncks -d time,"1861-01-01","1963-12-30" tos_Omon_HadGEM2-AO_historical_r1i1p1_186001-200512.nc ./tos_Omon_HadGEM2-AO_historical_r1i1p1_186101-196312_full.nc


ncks -d time,"1861-01-01","1963-12-30" HadGEM2-CC_1860-2005.nc  ./HadGEM2-CC_1860-1963.nc 


ncks -O --mk_rec_dmn time tos_Omon_HadGEM2-CC_historical_r1i1p1_185912-195911.nc HadGEM2-CC.nc
ncrcat -h HadGEM2-CC.nc tos_Omon_HadGEM2-CC_historical_r1i1p1_*.nc HadGEM2-CC_1860-2005.nc

ncks -O --mk_rec_dmn time SODA_1871.nc SODA_1871.nc
ncrcat -h ROC_SODA.nc SODA_*.nc FULL_SODA.nc

ncrcat -h tos_Omon_HadGEM2-CC_historical_r1i1p1_*.nc HadGEM2-CC_1860-2005.nc

ncrcat -h tos_Omon_HadGEM2-ES_historical_r1i1p1_*.nc HadGEM2-ES_1860-2005.nc

ncrcat -h tos_Omon_HadCM3_historical_r4i1p1_*.nc HadCM3_1860-2005.nc

ncrcat -h pottmp.*.nc GODAS_1980_2021.nc


ncks -O --mk_rec_dmn time SODA_300m_1871.nc SODA_300m_1871.nc
ncrcat -h RD.SODA_300m_1871.nc SODA_300m_*.nc SODAS_300m_1871_1973.nc

ncrcat -h RD.SODA_300m_1871.nc SODA_300m_*.nc SODAS_300m_1871_2000.nc


ncks -C -O -x -v lat_bnds,lon_bnds,time_bnds HadGEM2-CC_1860-2005.nc HadGEM2-CC_1860-2005_NOB.nc

cdo griddes HadGEM2-CC_1860-2005.nc > mygrid.txt

## article
https://ieeexplore.ieee.org/document/9001044

