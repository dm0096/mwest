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
import itertools as it
from haversine import haversine
from datetime import datetime
import os
import requests

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
    def download(filename, token, lat, lon, radius, max_stations, variables, start, end):
        API_ROOT = 'https://api.synopticdata.com/v2/'
        api_request_url = os.path.join(API_ROOT, "stations/timeseries")
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
    def jsonToDF(file):
        with open(file, 'r', encoding='utf-8', newline='') as f:
            loaded = json.load(f)
            dicts = [d for d in loaded['STATION']]
            df = pd.json_normalize(dicts)
        return df

        
    """"""
    def searchObservations(dic, keyName):
        for key in dic['OBSERVATIONS']:
            if key.startswith(keyName):
                return dic['OBSERVATIONS'][key]
            else:
                continue
        
    """"""
    def getStations(file):
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
    
    def processTS(self, variable, filt='60T'):
        # acquire times and format them
        timeForm = pd.to_datetime(self.time, infer_datetime_format=True)
        setattr(self, 'time', timeForm)
        # resample time and data with nearest neighbor from center
        df = Station.toDF(self, variable).resample(filt).nearest() 
        return df
    
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