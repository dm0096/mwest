# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 19:42:31 2020
Dean Meyer


"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dts
import pandas as pd
import json
import os
import requests
import itertools as it
from glob import glob
from haversine import haversine
from datetime import datetime
from operator import attrgetter


class Station:
    """ A class for wrangling station data """
    
    file = None
    
    """"""
    def __init__(self, 
                 name,
                 state,
                 elevation,
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
        self.elev = elevation
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
    def __add__(self, other):
        if self.name != other.name:
            raise ValueError('Stations are incommensurable')
        return Station(self.name, 
                       self.st, 
                       self.elev, 
                       self.mnet, 
                       self.lat, 
                       self.lon, 
                       np.concatenate([self.time, other.time]), 
                       np.concatenate([self.temp, other.temp]), 
                       np.concatenate([self.td, other.td]), 
                       np.concatenate([self.pres, other.pres]), 
                       np.concatenate([self.wspd, other.wspd]), 
                       np.concatenate([self.wdir, other.wdir]))
    
    """"""
    def download(filename, token, lat, lon, radius, max_stations, start, end):
        API_ROOT = 'https://api.synopticdata.com/v2/'
        api_request_url = os.path.join(API_ROOT, "stations/timeseries")
        variables = ('air_temp',
                     'dew_point_temperature',
                     'sea_level_pressure',
                     'wind_speed',
                     'wind_direction',
                     'precip_accum_one_minute')
        api_arguments = {'token':token,
                         'radius':(lat,lon,radius),
                         'limit':max_stations,
                         'vars':variables,
                         'start':start,
                         'end':end}
        req = requests.get(api_request_url, params=api_arguments)
        j = req.json()
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(j, f, ensure_ascii=False, indent=4)

    """"""
    def searchObservations(dic, keyName):
        for key in dic['OBSERVATIONS']:
            if key.startswith(keyName):
                values = dic['OBSERVATIONS'][key]
                if not isinstance(values[0], str):
                    return np.array(values, dtype=np.float)
                else:
                    return values
            else:
                continue
        
    """
    If a station has "None" instead of observations data, replace None with
    an array of NaNs whose length equals that of the existing data
    """
    def noneToNan(self):
        #find length of times
        maxLength = len(self.time)
        
        #generate array of NaNs
        nans = np.full_like(np.arange(maxLength), np.nan, dtype=float)
        
        if self.time is None:
            self.time = nans
        if self.temp is None:
            self.temp = nans
        if self.td is None:
            self.td = nans
        if self.pres is None:
            self.pres = nans
        if self.wspd is None:
            self.wspd = nans
        if self.wdir is None:
            self.wdir = nans
        return self
    
    """"""
    def loadJSON(file):
        with open(file, 'r', encoding='utf-8', newline='') as f:
            file = json.load(f)
            st = [Station(n['NAME'],
                          n['STATE'],
                          int(n['ELEVATION']),
                          int(n['MNET_ID']),
                          float(n['LATITUDE']),
                          float(n['LONGITUDE']),
                          Station.searchObservations(n, 'date_time'),
                          Station.searchObservations(n, 'air_temp'),
                          Station.searchObservations(n, 'dew_point'),
                          Station.searchObservations(n, 'pressure'),
                          Station.searchObservations(n, 'wind_speed'),
                          Station.searchObservations(n, 'wind_direction')) 
                    for n in file['STATION'] if 'OBSERVATIONS' in n]
            stNan = [s.noneToNan() for s in st]
        return stNan
    
    """"""
    def loadMultipleJSON(paths):
        allLoaded = [Station.loadJSON(p) for p in paths]
        concat = np.concatenate(allLoaded)
        get_attr = attrgetter('name')
        sorted_list = sorted(concat, key=get_attr)
        groupdict = {k: list(g) for k, g in it.groupby(sorted_list, get_attr)}
        #we can sum stations because of the __add__
        result = [np.sum(groupdict[g]) for g in groupdict]
        return result
    
    """"""
    def toDataFrame(self):
        data = {'time':self.time,
                'temp':self.temp,
                'td':self.td,
                'pres':self.pres,
                'wspd':self.wspd,
                'wdir':self.wdir}
        df = pd.DataFrame(data=data)
        df = df.set_index(pd.to_datetime(df['time']))
        df = df.drop(['time'], axis=1)
        df = df[~df.index.duplicated()]
        return df
    
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
    
    def rmse(predictions, targets):
        """Calculate root-mean-squared error"""
        return np.sqrt(((predictions - targets) ** 2).mean())
    
    def formatTime(self):
        """Format times into matplotlib-compatible floats"""
        dateTimes = [datetime.strptime(t, "%Y-%m-%dT%H:%M:%S%z") for t in self.time]
        self.time = dts.date2num(dateTimes)
    
    def toNearestStation(self, stations):
        """Return the station with the minimum haversine distance to a given 
        station and set 'haversine' attribute for the station with distance"""    
        h = [haversine((self.lat, self.lon), (s.lat, s.lon)) for s in stations]
        setattr(self, 'haversine', min(h))
        for s in stations:
            if haversine((self.lat, self.lon), (s.lat, s.lon)) == self.haversine:
                return s