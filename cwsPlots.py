import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
import numpy as np
from glob import glob
import pandas as pd

from station import Station

#%%

# two ways to load the json file
df = Station.jsonToDF('20191216.json')
#st = Station.getStations('20191216.json')
st = Station.getStations('201912310600_202001010600.json')

# cartopy
def cartopyMap():
    ax = plt.axes(projection=ccrs.PlateCarree())
    # counties
    reader = shpreader.Reader('C:/Users/Dean bean/Documents/capstone/counties/countyl010g.shp')
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

ax = cartopyMap()
# asos data
cwop = [s for s in st if s.mnet == 65]
lons = [a.lon for a in cwop]
lats = [a.lat for a in cwop]
ax.scatter(lons, lats, facecolor='r', edgecolor='k', marker='^', transform=ccrs.PlateCarree(), label='CWOP')

asos = [s for s in st if s.mnet == 1]
lons = [a.lon for a in asos]
lats = [a.lat for a in asos]
ax.scatter(lons, lats, c='b', marker='*', transform=ccrs.PlateCarree(), label='ASOS')
plt.legend()

# all data
#lons = [s.lon for s in st]
#lats = [s.lat for s in st]
#mnets = [s.mnet for s in st]
#ax.scatter(lons, lats, c=mnets, transform=ccrs.PlateCarree(), cmap='tab20')


#%%
#plot distances to nearest ASOS stations from CWOP stations
ax = cartopyMap()
nearest = [Station.toNearestStation(c, asos) for c in cwop]
#print([n.haversine for n in nearest])
for c in cwop:
    ax.scatter(c.lon, c.lat, marker='^', color='r')
for n in nearest:
    ax.scatter(n.lon, n.lat, marker='*', color='b')
for n in range(len(cwop)):
    ax.plot([cwop[n].lon, nearest[n].lon],[cwop[n].lat, nearest[n].lat])
legendElements = [Line2D([0], [0], marker='*', color='w', markerfacecolor='b', 
                         markersize=15, label='ASOS'),
                  Line2D([0], [0], marker='^', color='w', markerfacecolor='r', 
                         markersize=10, label='CWOP')]
ax.legend(handles=legendElements)
#plt.savefig('nearestCwopAsos_new.png', dpi=1000)

#%%
cwop = [s for s in st if s.mnet == 65]
nearest = [Station.toNearestStation(c, asos) for c in cwop]

n = 13

#temperature
#fig, ax = plt.subplots()
#Station.processTS(nearest[n], 'temp', filt='5T').plot(ax=ax, color='b')
#Station.processTS(cwop[n], 'temp', filt='5T').plot(ax=ax, color='r')


other = [s for s in st if s.name == 'HODGES']
nearest2 = [Station.toNearestStation(c, other) for c in other]
fig2, ax2 = plt.subplots()
Station.processTS(nearest[0], 'temp', filt='5T').plot(ax=ax2, color='b')
Station.processTS(other[0], 'temp', filt='5T').plot(ax=ax2, color='r', linestyle='--')
hodges = Station.processTS(other[0], 'temp', filt='5T') - 2.62 #estimated bias of 2.62
hodges.plot(ax=ax2, color='r')
plt.tight_layout()
#plt.savefig('hodgesErrorCorrected.png', dpi=1000)

#pressure
#fig2, ax2 = plt.subplots()
#Station.processTS(nearest[n], 'pres', filt='5T').plot(ax=ax2, color='b')
#Station.processTS(cwop[n], 'pres', filt='5T').plot(ax=ax2, color='r')

#wind speed
#fig3, ax3 = plt.subplots()
#x = Station.processTS(cwop[n], 'wspd', filt='5T').data
#asosTemps = Station.processTS(nearest[n], 'temp', filt='5T').data
#cwopTemps = Station.processTS(cwop[n], 'temp', filt='5T').data
#plt.scatter(x,np.abs(asosTemps-cwopTemps), color='r')

#for n in nearest:
#    Station.processTS(n, 'temp', filt='5T').plot(ax=ax, color='r')
#Station.processTS(cwop[0], 'temp', filt='5T').plot(ax=ax, color='b')

#plot corrected station temperaure based on "errors"

#%%
#read in multiple days of data and concatenate

st1 = Station.getStations('201912010600_201912020600.json')
st2 = Station.getStations('201912020600_201912030600.json')

stations = [st1, st2]

#find stations that are common between days
names = [[s.name for s in sts] for sts in stations]
set1 = map(set, names)
common_names = set.intersection(*set1)
filt1 = [s for s in st1 if s.name in common_names]
filt2 = [s for s in st2 if s.name in common_names]

#%%
#load all station days
paths = glob('*0600.json')
allStations = [Station.getStations(p) for p in paths]

#get names of stations if those stations appear every day of the month
numdays=31
names = [[s.name for s in sts] for sts in allStations]
namesFlat = [name for namesList in names for name in namesList]
uniqueNames = pd.unique(namesFlat)
counts = [namesFlat.count(u) for u in uniqueNames]
nameLenZip = list(zip(uniqueNames, counts))
namesDays = [z[0] for z in nameLenZip if z[1] == numdays]

#get all stations that have data every day of the month
allStationsMonth = [[s for s in st if s.name in namesDays] for st in allStations]

#remove stations if they don't contain enough unique observations
#generate a list of the number of unique temp observations 
listOfUnique = []
for s in allStationsMonth[0]:
    temps = np.array(s.temp, dtype=np.float)
    numOfUnique = len(np.unique(temps[~np.isnan(temps)]))
    listOfUnique = np.append(listOfUnique, numOfUnique)
print(listOfUnique)
#find indices where there are less than 5 unique observations
lessThan = np.where(listOfUnique < 5, False, True)
print(lessThan)
#remove stations that have less than 5 unique observations
allStationsMonth = [list(np.array(allSt)[lessThan]) for allSt in allStationsMonth]

#find range of dates with times always 21Z
numdays = 31
time = datetime(2019,12,1,21)
dtObjs = [time + timedelta(days=d) for d in range(numdays)]
datetimes = [o.strftime('%Y-%m-%d %H:%M:%S') for o in dtObjs]
#print(datetimes)

#potential temperature calculation
def thetaCalc(temp, pres):  #temp in C, pres in Pa
    return (temp + 273)*((100000.0/pres)**(2/7)) - 273    #returns theta in C

#pressure estimation from elevation and temp, hypsometric
def hypsometric(temp, elev):
    temp = temp + 273
    elev = elev * 0.3048
    H = 287 * temp / 9.81
    return 100000. * np.exp(-elev / H)

#
def plotAverages():
    return None
#%%
#trying to eliminate the big "for" loop below

#number of stations on a day of the month
numOfStationsOneDay = range(len(allStationsMonth[0]))
#lists groups of daily station data
station = [[s[i] for s in allStationsMonth] for i in numOfStationsOneDay]

#given a list of daily station data grouped by month, concatenate daily data
#into one month for each station. Then locate values at a given time of each day.
#Take the mean of those daily values at the given time. 
fig, ax = plt.subplots()

avgTemps = [pd.concat([Station.processTS(s, 'temp', filt='5T') for s in st]).at_time('21:00:00').data.mean() for st in station]
avgPress = [pd.concat([Station.processTS(s, 'pres', filt='5T') for s in st]).at_time('21:00:00').data.mean() for st in station]
thetas = thetaCalc(np.array(avgTemps), np.array(avgPress))

elevs = [s[0].elev for s in station]

mnets = [s[0].mnet for s in station]
asosInd = np.array(mnets) == 1
cwopInd = np.array(mnets) == 65
otherInd = (np.array(mnets) != 1) & (np.array(mnets) != 65)

plottingVar = avgTemps
ax.scatter(np.array(elevs)[asosInd], np.array(plottingVar)[asosInd], color='b', marker='*')
ax.scatter(np.array(elevs)[cwopInd], np.array(plottingVar)[cwopInd], color='r', marker='^')
ax.scatter(np.array(elevs)[otherInd], np.array(plottingVar)[otherInd], color='k', marker='P')
#plot a best fit line through the data
m, b = np.polyfit(elevs,avgTemps,1)
plt.plot(elevs, m*np.array(elevs) + b, 'k-')

#plot dry ALR
def dryALR(height):
    gamma = -9.8 #degrees C per km
    gamma = gamma / 3280.84 #km to feet
    return 12.75 + gamma*height - 700*gamma
dryALRhgts = np.arange(np.min(elevs), np.max(elevs), 2)
plt.plot(dryALRhgts, dryALR(dryALRhgts), 'k--') #plot dry ALR
#plt.plot([elevs, dryALRhgts],[avgTemps, dryALR(np.array(elevs))], 'r-') #plot errors

#legend
legendElements = [Line2D([0], [0], marker='*', color='w', markerfacecolor='b', 
                 markersize=15, label='ASOS'),
                  Line2D([0], [0], marker='^', color='w', markerfacecolor='r',
                         markersize=10, label='CWOP'),
                 Line2D([0], [0], marker='P', color='w', markerfacecolor='k',
                 markersize=10, label='Other'),
                Line2D([0],[0], linestyle='--', color='k', label='Dry ALR'),
                Line2D([0],[0], linestyle='-', color='k', label='Fit')]
ax.legend(handles=legendElements)
plt.title('Average temperature at ' + str(time.hour) + 'Z for ' 
          + str(time.strftime('%B %Y')))
plt.ylabel('Temperature ($^\circ C$)') #temperature
plt.xlabel('Elevation $(ft)$')
plt.tight_layout()
#plt.savefig('monthTempDecember21Z.png', dpi=1000)


#value differences between average temps and the dry ALR model
errors = np.array(avgTemps) - dryALR(np.array(elevs))

#store errors from dry ALR model in each station as a new attribute
stations = [s[0] for s in station]
for i in range(len(station)): #len 69
    setattr(stations[i], 'error', errors[i])

#load data from a random day, maintaining stored errors
st1 = Station.getStations('20191216.json')
stationNames = [s.name for s in stations]
st1 = [s for s in st1 if s.name in stationNames]
for i in range(len(st1)): #inherit error attribute if station names match 
    for s in stations:
        if st1[i].name == s.name:
            setattr(st1[i], 'error', s.error)
        else:
            continue
    
#correct data from a random day with stored errors
decatur = Station.processTS(st1[0], 'temp', filt='5T')
decaturCorr = Station.processTS(st1[0], 'temp', filt='5T') - st1[0].error

#plot corrected data
fig, ax2 = plt.subplots()
decatur.plot(ax=ax2, linestyle='--')
decaturCorr.plot(ax=ax2)

#%%
#potential temperature analysis, monthly average as above

#find thetas with hypsometric equation
elevs = [s[0].elev for s in station]
press = hypsometric(np.array(avgTemps), np.array(elevs)) 
thetas = thetaCalc(np.array(avgTemps), press)

#plot data
plottingVar = thetas
fig, ax = plt.subplots()
ax.scatter(np.array(elevs)[asosInd], np.array(plottingVar)[asosInd], color='b', marker='*')
ax.scatter(np.array(elevs)[cwopInd], np.array(plottingVar)[cwopInd], color='r', marker='^')
ax.scatter(np.array(elevs)[otherInd], np.array(plottingVar)[otherInd], color='k', marker='P')

#linear fit through data
m, b = np.polyfit(elevs,thetas,1)
plt.plot(elevs, m*np.array(elevs) + b, 'k-')

#dry ALR through average of data at 700 ft
dryALRhgts = np.arange(np.min(elevs), np.max(elevs), 2)
plt.plot([np.min(elevs), np.max(elevs)], [14.75, 14.75], 'k--')

#legend
legendElements = [Line2D([0], [0], marker='*', color='w', markerfacecolor='b', 
                 markersize=15, label='ASOS'),
                  Line2D([0], [0], marker='^', color='w', markerfacecolor='r',
                         markersize=10, label='CWOP'),
                 Line2D([0], [0], marker='P', color='w', markerfacecolor='k',
                 markersize=10, label='Other'),
                Line2D([0],[0], linestyle='--', color='k', label='Dry ALR'),
                Line2D([0],[0], linestyle='-', color='k', label='Fit')]
ax.legend(handles=legendElements)

#title, axes labels
plt.title('Average $\Theta$ at ' + str(time.hour) + 'Z for ' 
          + str(time.strftime('%B %Y')))
plt.ylabel('Potential temperature, $\Theta$ ($^\circ C$)') #temperature
plt.xlabel('Elevation $(ft)$')
plt.tight_layout()
plt.savefig('monthThetaDecember_21_hypso.png', dpi=1000)

#%%

#do boxplot analysis with corrected data

#apply errors
dfs = [Station.processTS(s, 'temp', filt='5T') - s.error for s in st1]

#get data for each station at 12, 15, 18, 21, 00 Z
t12 = np.array([d.at_time('12:00:00').data for d in dfs]).flatten()
t12 = t12[~np.isnan(t12)]
t15 = np.array([d.at_time('15:00:00').data for d in dfs]).flatten()
t18 = np.array([d.at_time('18:00:00').data for d in dfs]).flatten()
t21 = np.array([d.at_time('21:00:00').data for d in dfs]).flatten()
#t00 = np.array([d.at_time('00:00:00').data for d in dfs]).flatten()
#t00 = t00[~np.isnan(t00)]

#plot boxplots
#fig, ax = plt.subplots(figsize=[8,4])
#boxplots = plt.boxplot(t18)
#boxplots = plt.boxplot([t12, t15, t18, t21, t00], labels=['12','15','18','21', '00'])
boxplots = plt.boxplot([t12, t15, t18, t21], labels=['12','15','18','21'])

#plot values for caps, IQR, median
#caps
capVals = np.array([np.round(item.get_ydata()[0], 2) for item in boxplots['caps']])
capX = np.array([item.get_xdata()[1] for item in boxplots['caps']])
for i in range(len(capX)):
    plt.text(capX[i]+0.1, capVals[i], capVals.astype(str)[i])
#IQR
iqrVals = np.array([np.round(item.get_ydata()[1:3], 2) for item in boxplots['boxes']]).flatten()
iqrX = np.array([item.get_xdata()[1:3] for item in boxplots['boxes']]).flatten()
for i in range(len(capX)):
    plt.text(iqrX[i]+0.1, iqrVals[i], iqrVals.astype(str)[i])
    
plt.title('12/16/2019 5-min temperature, corrected stations')
plt.xlabel('UTC')
plt.ylabel('Temperature ($^\circ C$)')
plt.tight_layout()
#plt.savefig('161219_box_temp_5min_corrected.png', dpi=1000)

#%%

#do time series of corrected temp data 
fig, ax = plt.subplots()
for d in dfs:
    d.plot(ax=ax, legend=False, color='k')
#plt.savefig('161219_temps_corr.png', dpi=1000)

#%%

#plot station data for a time on a map with radar data
fig = plt.figure(figsize=[15, 7])
ax = cartopyMap()

import pyart
path = 'C:/Users/Dean bean/Documents/capstone/20191216_khtx/'
fname = 'KHTX20191216_231258_V06.ar2v'
file = path+fname
radar = pyart.io.read(file)
display = pyart.graph.RadarMapDisplay(radar)
#display.set_limits(xlim=[-100, 100], ylim=[-100,100])
#display.plot('reflectivity', 0, vmin=-32, vmax=64.)

lons = [s.lon for s in st1]
lats = [s.lat for s in st1]

display.plot_ppi_map('reflectivity', 0, vmin=-32, vmax=64., 
                     min_lon=np.min(lons)-0.25, max_lon=np.max(lons)+0.25,
                     min_lat=np.min(lats)-0.25, max_lat=np.max(lats)+0.25,
                     projection=ccrs.PlateCarree(), ax=ax, cmap='pyart_Carbone42')

#plot colored dots based on temperature
#scatter = ax.scatter(lons, lats, c=t18, cmap='cool', edgecolors='k')
#plt.colorbar(scatter)

#temperature values at a time
t22 = np.array([d.at_time('23:15:00').data for d in dfs]).flatten()

#plot markers based on mnet ID with annotated temperature values
mnets = [s.mnet for s in st1]
asosInd = np.array(mnets) == 1
cwopInd = np.array(mnets) == 65
otherInd = (np.array(mnets) != 1) & (np.array(mnets) != 65)
ax.scatter(np.array(lons)[asosInd], np.array(lats)[asosInd], color='b', marker='*')
ax.scatter(np.array(lons)[cwopInd], np.array(lats)[cwopInd], color='r', marker='^')
ax.scatter(np.array(lons)[otherInd], np.array(lats)[otherInd], color='k', marker='P')
#for i in range(len(st1)):
#    plt.text(lons[i], lats[i], str(np.round(t22[i], 1)))

#temperature contour
contour = ax.tricontour(lons, lats, t22, 5, colors='k')
ax.clabel(contour, inline=1, fontsize=10, fmt='%1.1f')

plt.savefig('121619_2315_temp_contour.png', dpi=1000)
    
#%%

#contour plot of temperatures
t23 = np.array([d.at_time('23:15:00').data for d in dfs]).flatten()
fig, ax = plt.subplots(figsize=[10,10])
contour = ax.tricontour(lons, lats, t23, 5, colors='k')
ax.clabel(contour, inline=1, fontsize=10)
#plt.colorbar(contour)