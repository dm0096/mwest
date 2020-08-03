import requests
import os
import json

API_ROOT = 'https://api.synopticdata.com/v2/'
API_TOKEN = 'f3d9fbafeada4687b51e937a8830dec3'
api_request_url = os.path.join(API_ROOT, "stations/timeseries")
start = 201912310600
end = 202001010600
api_arguments = {'token':API_TOKEN,
#                 'radius':(34.582253,-86.927738,50),
                 'radius':(34.653603,-86.943709,50),
#                 'network':'APRSWXNET/CWOP',
                 'limit':100,
                 'vars':('air_temp','dew_point_temperature','relative_humidity','pressure','sea_level_pressure','wind_speed','wind_direction'),
                 'start':start,
                 'end':end}
req = requests.get(api_request_url, params=api_arguments)
j = req.json()
with open((str(start)+ '_' + str(end) + '.json'), 'w', encoding='utf-8') as f:
    json.dump(j, f, ensure_ascii=False, indent=4)