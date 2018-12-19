import numpy as np
import datetime as dt
from netCDF4 import Dataset
import pickle 

# Load one of the nc files to get list of coordinates
data = Dataset('humidity/HadISDH.landq.4.0.0.2017f_FLATgridIDPHA5by5_anoms8110_MAR2018_cf.nc')
latitude = np.ma.filled(data.variables['latitude'])
longitude = np.ma.filled(data.variables['longitude'])

# Store coordinates (lat, long) in a list
lat_long = []

for lat in latitude:
    for long in longitude:
        lat_long.append((lat, long))

# Load the time: The numbers are the days since 1/1/1973. Range is 1/1/1973 to 12/1/2017. 
time = np.ma.filled(data.variables['time'])
# Load monthly mean absolute humidity expressed in g/kg
# This is the specific humidity, the ratio of the mass of water vapour to the mass of moist air
humidity = np.ma.filled(data.variables['q_abs'])

humidity_grid = {}

# Define a variable for the reference date, 1/1/1973
reference_date = dt.datetime(1973,1,1)

for i in range(len(time)):
    # Get datetime for the date
    date = reference_date + dt.timedelta(int(time[i]))

    # Key is (year, month)
    key = (date.year, date.month)

    #humidity is (t, lat, lon), where t is the months
    month_humidities = humidity[i].ravel()

    humidity_grid[key] = {}

    for j in range(len(lat_long)):
        # Look only at the latitudes and longitudes around the United States
        if (lat_long[j][0] <= 72.5) and (lat_long[j][0] >= 17.5) \
        and (lat_long[j][1] <= -64.5) and (lat_long[j][1] >= -179.5):
            humidity_grid[key][lat_long[j]] = month_humidities[j]

# Save dictionary into a file
with open('us_humidity.pickle', 'wb') as fp:
    pickle.dump(humidity_grid, fp, protocol = pickle.HIGHEST_PROTOCOL)
