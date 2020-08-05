import numpy as np
import json
import matplotlib.pyplot as plt
import itertools as it
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

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

# Reads .json ASOS/RAWS data from Synoptic
def readJSON(file):
    with open(file, 'r', encoding='utf-8', newline='') as f:
        dataset = json.load(f)
    return dataset

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
    # resample time and data with median filter
    dfs = [toDF(timeForm[i], data[i]).resample(filt).median() for i in range(len(timeForm))]
    return dfs

# convert temperature in deg C to potential temperature with pres in Pa
def theta(temp, pres):
    return temp*((100000.0/pres)**(2/7))

# calculate RMSE
# find the mean of ASOS data
def meansDF(dfs):
    concat = pd.concat(dfs)
    by_row_idx = concat.groupby(concat.index)
    means = by_row_idx.mean()
    return means

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def cartopyMap(path):
    ax = plt.axes(projection=ccrs.PlateCarree())
    # counties
    reader = shpreader.Reader(path)
    counties = list(reader.geometries())
    COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())
    # add the features
    ax.add_feature(cfeature.LAND) 
    ax.add_feature(cfeature.LAKES)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)
    ax.add_feature(COUNTIES, facecolor='none', edgecolor='gray')
    ax.coastlines()
    return ax

#potential temperature calculation
def thetaCalc(temp, pres):  #temp in C, pres in Pa
    return (temp + 273)*((100000.0/pres)**(2/7)) - 273    #returns theta in C

#pressure estimation from elevation and temp, hypsometric equation
def hypsometric(temp, elev):
    temp = temp + 273
    elev = elev * 0.3048
    H = 287 * temp / 9.81
    return 100000. * np.exp(-elev / H)

#dry adiabatic lapse rate
def dryALR(height):
    gamma = -9.8 #degrees C per km
    gamma = gamma / 3280.84 #km to feet
    return 12.75 + gamma*height - 700*gamma

#getting time series data at a specific time from a DataFrame
def getAtTime(df, stamp, col):
    try:
        data = df.loc[stamp][col]
        return data
    except:
        data = np.nan
        return data