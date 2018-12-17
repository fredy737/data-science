import numpy as np
import datetime as dt
from netCDF4 import Dataset
import pickle 

# Load one of the nc files to get list of coordinates
data = Dataset('Land_and_Ocean_LatLong1.nc')
latitude = np.ma.filled(data.variables['latitude'])
longitude = np.ma.filled(data.variables['longitude'])

# Store coordinates (lat, long) in a list
lat_long = []

for lat in latitude:
    for long in longitude:
        lat_long.append((lat, long))

def is_leap(year):
    if year % 4 != 0:
        return 0
    elif year % 100 != 0:
        return 1
    elif year % 400 != 0:
        return 0
    else:
        return 1

def convert_datetime(number):
    year = int(number)
    d = dt.timedelta(days = (number - year) * (365 + is_leap(year)))
    day_one = dt.datetime(year, 1, 1)
    date = d + day_one

    return date

def load_climatology(climatology):
    # climatology = (12, lat, long) array, where 12 is the month
    climatology_dict = {}

    for i in range(climatology.shape[0]):
        # i is the index for the month (0 is January, 11 is December)
        key = i + 1

        climatology_dict[key] = {}

        # Get climatology array for the month. Unravel the 2D array into a 1D array
        climatology_month = climatology[i].ravel()

        for j in range(len(lat_long)):
            climatology_dict[key][lat_long[j]] = climatology_month[j]

    return climatology_dict

# Load the time (1850 to 2018.5...). Need to convert decimals to months and day
time = np.ma.filled(data.variables['time'])
temperatures = np.ma.filled(data.variables['temperature'])
climatology = np.ma.filled(data.variables['climatology'])

# Create dictionary where the keys are month containing climatology values in the subgrid
climatology_dict = load_climatology(climatology)

temp_grid = {}

for i in range(len(time)):
    # Get datetime for the date
    date = convert_datetime(time[i])

    # Only look at 1992 and beyond
    if date.year >= 1992:
        # Key is (year, month)
        key = (date.year, date.month)

        month_temperatures = temperatures[i].ravel()

        temp_grid[key] = {}

        for j in range(len(lat_long)):
            # Look only at the latitudes and longitudes around the Santa Cruz country in California
            if (lat_long[j][0] <= 70.5) and (lat_long[j][0] >= 17.5) \
            and (lat_long[j][1] <= -64.5) and (lat_long[j][1] >= -179.5):
                temp_grid[key][lat_long[j]] = climatology_dict[date.month][lat_long[j]] + month_temperatures[j]

# Save dictionary into a file
with open('us_month_land_and_sea.pickle', 'wb') as fp:
    pickle.dump(temp_grid, fp, protocol = pickle.HIGHEST_PROTOCOL)
