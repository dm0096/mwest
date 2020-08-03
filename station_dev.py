# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 19:42:31 2020
Dean Meyer

An attempt to read and plot 1-minute ASOS and CWS data
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import itertools as it
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature 

"""
Function for reading ASOS 64050 .dat files
"""
def read6405(fin):
    dtype = [('wban', int),
             ('id', 'U9'), 
             ('id2', 'U3'),
             ('year', 'U4'),
             ('month', 'U2'),
             ('day', 'U2'),
             ('ltime', 'U4'),
             ('utime', 'U4'),
             ('ext', float),
             ('night', 'U1'), 
             ('wdir2', int), 
             ('wspd2', int), 
             ('wdir5', int), 
             ('wspd5', int), 
             ]
    data = np.genfromtxt(fin, 
                         delimiter=[5,4,4,4,2,2,4,4,9,3,33,5,5,5], 
                         dtype=dtype, 
                         usecols=[1,3,4,5,6,7,8,9,10,11,12,13],
                         names=None,    #allows indexing by names of fields
                         missing_values=['M','[',']','[   M  ]'],
                         autostrip=True)
    return data

"""
Function for reading ASOS 64060 .dat files
"""
def read6406(fin):
    dtype = [('wban', int),
             ('id', 'U9'), 
             ('id2', 'U3'),
             ('year', 'U4'),
             ('month', 'U2'),
             ('day', 'U2'),
             ('ltime', 'U4'),
             ('utime', 'U4'),
             ('pid', 'U2'),
             ('unk', 'U4'), 
             ('precip', float), 
             ('freq', 'U5'), 
             ('p1', float), 
             ('p2', float),
             ('t', int),
             ('td', int)
             ]
    data = np.genfromtxt(fin, 
                         delimiter=[5,4,4,4,2,2,4,4 ,5,6,8,19,9,8,13,5], 
                         dtype=dtype, 
                         usecols=[1,3,4,5,7,8,10,12,13,14,15],
                         names=None,    #allows indexing by names of fields
                         missing_values=['M','[',']','[   M  ]'],
                         autostrip=True)
    return data

def thetavcalc(
        probe):
    # Temp and dewpoint in C
    Td_C = (probe['td'] - 32.) * (5/9) 
    T_C = (probe['t'] - 32.) * (5/9) 
    # Potential temp
    theta = T_C * (1000.0 / probe['p1']) ** (287.0 / 1004.0)
    # Vapor pressure
#    vap_P = 6.11 * np.exp((2500000.0 / 461.0) * (273.15 ** -1 - Td_K ** -1.0))
    vap_P = 0.6112 * np.exp((17.67 * Td_C) / (Td_C + 243.5))
    # Mixing ratio
    MR = .622 * vap_P / (probe['p1'] - vap_P)  

    theta_V = theta * (1.0 + .61 * MR)
    return theta_V

""" Reads .json ASOS/RAWS data from Synoptic """
def readJSON(file):
    with open(file, 'r', encoding='utf-8', newline='') as f:
        dataset = json.load(f)
    return dataset

####Testing
#file = readJSON('C:/Users/Dean bean/Documents/capstone/20191216.json')
#
#isactive = [file['STATION'][n]['STATUS'] == 'ACTIVE' for n in range(len(file['STATION']))]
#station = [file['STATION'][n] for n in range(len(file['STATION'])) if isactive[n]]
#
#lat = [float(station[n]['LATITUDE']) for n in range(len(station))]
#lon = [float(station[n]['LONGITUDE']) for n in range(len(station))]
#
#ax = plt.axes(projection=ccrs.PlateCarree())
#
#ax.add_feature(cfeature.LAND) 
#ax.add_feature(cfeature.LAKES)
#ax.add_feature(cfeature.RIVERS)
#ax.add_feature(cfeature.BORDERS)
#ax.add_feature(cfeature.STATES)
#ax.coastlines()
#
#ax.scatter(lon,lat,transform=ccrs.PlateCarree())
#plt.show()

""" A class for wrangling with station data """
class Station:
    
    file = None
    
    """"""
    def __init__(self, 
                 name,
                 state,
                 mnet_id,
                 latitude,
                 longitude,
                 time, 
                 temperature,
                 dewpoint, 
                 pressure, 
                 windSpeed, 
                 windDirection):
        self.name = name
        self.st = state
        self.mnet = mnet_id
        self.lat = latitude
        self.lon = longitude
        self.time = time
        self.temp = temperature
        self.td = dewpoint
        self.pres = pressure
        self.wspd = windSpeed
        self.wdir = windDirection

    """"""
    def jsonToDF(file):
        with open(file, 'r', encoding='utf-8', newline='') as f:
            loaded = json.load(f)
            dicts = [d for d in loaded['STATION']]
            df = pd.json_normalize(dicts)
        return df

        
    """"""
    def stationData(dic, keyName):
        for key in dic['OBSERVATIONS']:
            if key.startswith(keyName):
                return dic['OBSERVATIONS'][key]
            else:
                continue
        
    """"""
    @classmethod
    def getStations(cls, file):
        st = [Station(n['NAME'],
                      n['STATE'],
                      int(n['MNET_ID']),
                      float(n['LATITUDE']),
                      float(n['LONGITUDE']),
                      Station.stationData(n, 'date_time'),
                      Station.stationData(n, 'air_temp'),
                      Station.stationData(n, 'dew_point'),
                      Station.stationData(n, 'sea_level_pressure'),
                      Station.stationData(n, 'wind_speed'),
                      Station.stationData(n, 'wind_direction')) 
                for n in file['STATION'] if 'OBSERVATIONS' in n]
        return st
    
    """"""
    def plotStation(st, attr):
        ids = [getattr(s, attr) for s in st]
        gp = [[s for s in st if s.mnet == n] for n in np.unique(ids)] #list of stations grouped by attr
        markers = it.cycle(["." , "," , "o" , "v" , "^" , "<", ">"])
        fig, ax = plt.subplots()
        for ls in gp:
            lons = [s.lon for s in ls]
            lats = [s.lat for s in ls]
            values = [getattr(s, attr) for s in ls]
            ax.scatter(lons, lats, c=values, marker=next(markers))
        return None
    
    """"""
    def toDF(self, attr):
        frame = pd.DataFrame(self.time, columns=['datetime'])
        frame['data'] = getattr(self, attr)
        return frame.set_index('datetime')
    
# File reading example
#f=Station.readJSON('20191216.json')
#st=Station.getStations(f)
#print(st)

# Plotting example
#fig, ax = plt.subplots(figsize=(7,7))
#lons = [s.lon for s in st]
#lats = [s.lat for s in st]
#ids = [s.mnet for s in st]
#scatter=ax.scatter(lons, lats, c=ids, cmap='tab20', marker='D')
#ax.legend(*scatter.legend_elements(num=None))

# Markers for unique plots
#import itertools as it
#markers = it.cycle(["." , "," , "o" , "v" , "^" , "<", ">"])
#colors = it.cycle(['r','g','b','c','m', 'y', 'k'])
#
#gp = [[s for s in st if s.mnet == n] for n in np.unique(ids)] #list of stations grouped by mnet id



#Station.plotStation(st, 'mnet')

#%%

#f = Station.readJSON('20191216.json')
#dicts = [d for d in f['STATION']]
#df = pd.json_normalize(dicts)

df = Station.jsonToDF('20191216.json')

def drawStations(lons, lats, times, data):
    markers = it.cycle(["." , "," , "o" , "v" , "^" , "<", ">"])
    fig, ax = plt.subplots()
    ax.scatter(lons, lats, c=data, marker=next(markers))
    return None

def getLats(df):
    return df.loc[0, 'LATITUDE'].astype(float)

def getLons(df):
    return df.loc[0, 'LONGITUDE'].astype(float)

def getTimes(df):
    return df.loc[0, 'OBSERVATIONS.date_time'][0]

def getData(df):
    return df.loc[0, 'OBSERVATIONS.air_temp_set_1'][0]

def getStuff(df):
    sl = np.arange(7,50)
#    sl = range(len(df))
    return {'lons': [float(df.loc[n, 'LONGITUDE']) for n in sl],
            'lats': [float(df.loc[n, 'LATITUDE']) for n in sl],
            'times': [df.loc[n, 'OBSERVATIONS.date_time'][0] for n in sl],
            'data': [df.loc[n, 'OBSERVATIONS.air_temp_set_1'][0] for n in sl]}

#getData(df)
#drawStations(getLats(df), getLons(df), getTimes(df), getData(df))
drawStations(**getStuff(df))

#%%

# Plotting two ASOS time series together on same timeline
asos = df.groupby('MNET_ID').groups['1']
dfAsos = df.iloc[asos]
t1 = dfAsos['OBSERVATIONS.date_time'].iloc[0]
t2 = dfAsos['OBSERVATIONS.date_time'].iloc[1]
t1f = pd.to_datetime(t1, infer_datetime_format=True)
t2f = pd.to_datetime(t2, infer_datetime_format=True)
tp1 = dfAsos['OBSERVATIONS.air_temp_set_1'].iloc[0]
tp2 = dfAsos['OBSERVATIONS.air_temp_set_1'].iloc[1]
fig, ax = plt.subplots()
ax.plot(t1f, tp1)
ax.plot(t2f, tp2)
plt.show()

#%%
# create dataframes for every station
def toDF(time, data):
    frame = pd.DataFrame(time, columns=['datetime'])
    frame['data'] = data
    return frame.set_index('datetime')

# process time series data for plotting
def processTS(df, variable, filt='60T', mnet=None):
    # acquire stations by their MNET ID
    if mnet:
        df = df.iloc[df.groupby('MNET_ID').groups[mnet]]
    # acquire times and format them
    time = [n for n in df['OBSERVATIONS.date_time'] if isinstance(n, list)]
    timeForm = [pd.to_datetime(l, infer_datetime_format=True) for l in time]
    # acquire data from DF
    data = [n for n in df[variable] if isinstance(n, list)]
    # resample time and data with nearest neighbor from center
    dfs = [toDF(timeForm[i], data[i]).resample(filt).nearest() for i in range(len(timeForm))]
    return dfs

# convert temperature in deg C to potential temperature with pres in Pa
def theta(temp, pres):
    return temp*((100000.0/pres)**(2/7))

#%%
# ASOS
dfsAsos = processTS(df, 'OBSERVATIONS.air_temp_set_1', '5T', '1')
fig, ax = plt.subplots()
for d in dfsAsos:
    ax.plot(d.index, d.data, color='b')
plt.title('5-min ASOS Temperature')
plt.xlabel('UTC')
plt.ylabel('Temperature (C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('161219_asos_5min.png', dpi=1000)

#%%
# CWOP
dfsCwop = processTS(df, 'OBSERVATIONS.air_temp_set_1', '5T', '65')
fig, ax = plt.subplots()
for d in dfsCwop:
    ax.plot(d.index, d.data, color='g')
plt.title('5-min APRSWXNet/CWOP Temperature')
plt.xlabel('UTC')
plt.ylabel('Temperature (C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('161219_cwop_5min.png', dpi=1000)

#%%
# Soil Climate Analysis (SCAN)
dfsScan = processTS(df, 'OBSERVATIONS.air_temp_set_1', '5T', '29')
fig, ax = plt.subplots()
for d in dfsScan:
    ax.plot(d.index, d.data, color='r')
plt.title('5-min SCAN Temperature')
plt.xlabel('UTC')
plt.ylabel('Temperature (C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('161219_scan_5min.png', dpi=1000)

#%%
# plot all three!
fig, ax = plt.subplots()
for d in dfsCwop:
    ax.plot(d.index, d.data, color='g', label='APRSWXNet/CWOP')
for d in dfsAsos:
    ax.plot(d.index, d.data, color='b', label='ASOS')
for d in dfsScan:
    ax.plot(d.index, d.data, color='r', label='SCAN')
plt.title('60-min ASOS, APRSWXNet/CWOP, SCAN Temperature')
plt.xlabel('UTC')
plt.ylabel('Temperature (C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('161219_asosCwopScan.png', dpi=1000)

#%%
# plot histogram of temperature at 18 UTC
# create DFs for all groups
dfs = processTS(df, 'OBSERVATIONS.air_temp_set_1', '60T')
# slice data based on one time
tSlice = [n.loc['2019-12-16 18:00:00'].data for n in dfs if '2019-12-16 18:00:00' in n.index]
tSlice = np.array(tSlice)
tSlice = tSlice[~np.isnan(tSlice)]
# plot histogram
plt.hist(tSlice)
plt.title('18 UTC Temperature')
plt.xlabel('Temperature (C)')
plt.ylabel('Frequency')
plt.tight_layout()
#plt.savefig('161219_18_hist.png')

#%%
# plot boxplots of temperature at different times
# create DFs for all groups
dfs = processTS(df, 'OBSERVATIONS.air_temp_set_1', '5T')
# slice data based on one time
t18 = np.array([n.loc['2019-12-16 18:00:00'].data 
          for n in dfs if '2019-12-16 18:00:00' in n.index])
t18 = t18[~np.isnan(t18)]
t12 = np.array([n.loc['2019-12-16 12:00:00'].data 
          for n in dfs if '2019-12-16 12:00:00' in n.index])
t12 = t12[~np.isnan(t12)]
t15 = np.array([n.loc['2019-12-16 15:00:00'].data 
          for n in dfs if '2019-12-16 15:00:00' in n.index])
t15 = t15[~np.isnan(t15)]
t21 = np.array([n.loc['2019-12-16 21:00:00'].data 
          for n in dfs if '2019-12-16 21:00:00' in n.index])
t21 = t21[~np.isnan(t21)]
t00 = np.array([n.loc['2019-12-17 00:00:00'].data 
          for n in dfs if '2019-12-17 00:00:00' in n.index])
t00 = t00[~np.isnan(t00)]
# plot boxplots
boxplots = plt.boxplot([t12, t15, t18, t21, t00], labels=['12','15','18','21', '00'])
plt.title('12/16/2019 5-min Temperature, All Stations')
plt.xlabel('UTC')
plt.ylabel('Temperature (C)')
plt.tight_layout()
plt.savefig('161219_box_temp_5min_nearest.png', dpi=1000)

#%%
# plot boxplots of wind speed at different times
# create DFs for all groups
dfs = processTS(df, 'OBSERVATIONS.wind_speed_set_1', '60T')
# slice data based on one time
t18 = np.array([n.loc['2019-12-16 18:00:00'].data 
          for n in dfs if '2019-12-16 18:00:00' in n.index])
t18 = t18[~np.isnan(t18)]
t12 = np.array([n.loc['2019-12-16 12:00:00'].data 
          for n in dfs if '2019-12-16 12:00:00' in n.index])
t12 = t12[~np.isnan(t12)]
t15 = np.array([n.loc['2019-12-16 15:00:00'].data 
          for n in dfs if '2019-12-16 15:00:00' in n.index])
t15 = t15[~np.isnan(t15)]
t21 = np.array([n.loc['2019-12-16 21:00:00'].data 
          for n in dfs if '2019-12-16 21:00:00' in n.index])
t21 = t21[~np.isnan(t21)]
t00 = np.array([n.loc['2019-12-17 00:00:00'].data 
          for n in dfs if '2019-12-17 00:00:00' in n.index])
t00 = t00[~np.isnan(t00)]
# plot boxplots
boxplots = plt.boxplot([t12, t15, t18, t21, t00], labels=['12','15','18','21', '00'])
plt.title('12/16/2019 Wind Speed, All Stations')
plt.xlabel('UTC')
plt.ylabel('Wind Speed (m/s)')
plt.tight_layout()
#plt.savefig('161219_box_temp.png')

#%%
# calculate RMSE
# find the mean of ASOS data
def meansDF(dfs):
    concat = pd.concat(dfs)
    by_row_idx = concat.groupby(concat.index)
    means = by_row_idx.mean()
    return means

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

fig, ax = plt.subplots()
#for d in dfsAsos:
#    ax.plot(d.index, d.data, color='b')
meansAsos = meansDF(dfsAsos)
meansCwop = meansDF(dfsCwop)
meansScan = meansDF(dfsScan)
ax.plot(meansAsos, color='b', label='ASOS')
ax.plot(meansCwop, color='r', label='APRSWXNet/CWOP')
ax.plot(meansScan, color='g', label='SCAN')
ax.legend()

rmseCwop = rmse(meansCwop, meansAsos).data #RMSE btwn CWOP and ASOS
rmseScan = rmse(meansScan, meansAsos).data #RMSE btwn SCAN and ASOS
plt.text(0.1, 0.1, ('RMSE: ' + str(rmseCwop)), color='r', transform=ax.transAxes)
plt.text(0.1, 0.05, ('RMSE: ' + str(rmseScan)), color='g', transform=ax.transAxes)

#plt.title('60-min ASOS, APRSWXNet/CWOP, SCAN Temperature')
plt.xlabel('UTC')
plt.ylabel('Temperature (C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('161219_means.png', dpi=1000)

#%%
# representing RMSE and stations on a map

# run the cell before this one!!

