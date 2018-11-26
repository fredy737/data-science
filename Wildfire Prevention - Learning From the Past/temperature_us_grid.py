import numpy as np
from netCDF4 import Dataset
import datetime as dt
import pickle

# Load one of the nc files to get list of coordinates
data_lat_long = Dataset('daily_land/Complete_TAVG_Daily_LatLong1_1990.nc')
latitude = np.ma.filled(data_lat_long.variables['latitude'])
longitude = np.ma.filled(data_lat_long.variables['longitude'])

# Store coordinates (lat, long) in a list
lat_long = []

for lat in latitude:
    for long in longitude:
        lat_long.append((lat, long))

# Create a dictionary: dict[(month, day)] = {dictionary where keys are (lat, long)}
def load_climatology(climatology):
    # climatology = (365, lat, long) array, where 365 is day of year
    # Initialize the dictionary
    climatology_dict = {}
    
    for i in range(climatology.shape[0]):
        # Convert i = (0, 364) into datetime objects in an arbitrary non-leap year, like 1991
        day_of_year = dt.datetime(year = 1991, month = 1, day = 1) + dt.timedelta(i)
        # Define key and empty dictionary for the dictionary
        key = (day_of_year.month, day_of_year.day)
        climatology_dict[key] = {}
        
        # Get climatology array for the day of year. Unravel the 2D array into a 1D array
        climatology_doy = climatology[i].ravel()
        
        for j in range(len(lat_long)):
            climatology_dict[key][lat_long[j]] = climatology_doy[j]
            
    return climatology_dict

def load_temperature_grid(filename):
    # Read the nc file into data
    data = Dataset(filename)
    
    year = np.ma.filled(data.variables['year'])
    month = np.ma.filled(data.variables['month'])
    day = np.ma.filled(data.variables['day'])
    
    # Load the temperature matrix (days, lat, long)
    temperatures = np.ma.filled(data.variables['temperature'])
    
    # climatology: an estimate of the true surface temperature for each day of year (1-365) during 
    # the period January 1951 to December 1980 reported in degrees C. Treat as average.
    # Final temperature will be climatology + temperature (anomaly) = recorded temperature for each grid square
    climatology = np.ma.filled(data.variables['climatology'])
    
    # Create a dictionary where the keys are (month, day) containing climatology vales in the subgrid
    climatology_dict = load_climatology(climatology)

    date = np.vstack((year, month, day)).T
            
    temp_grid = {}

    for i in range(date.shape[0]):
        # Get 1D array of temperatures in the grid for the day
        day_temperatures = temperatures[i].ravel()

        key = (int(date[i, 0]), int(date[i, 1]), int(date[i, 2]))

        temp_grid[key] = {}

        for j in range(len(lat_long)):
            # Look only at latitutdes and longitudes in the subgrid containing California
            if (lat_long[j][0] <= 70.5) and (lat_long[j][0] >= 17.5) \
            and (lat_long[j][1] <= -64.5) and (lat_long[j][1] >= -172.5):
                # Address leap days. Use average value of February 28 and March 1 for climatology value
                if int(date[i, 1]) == 2 and int(date[i, 2]) == 29:
                    day_climatology = .5 * (climatology_dict[(2, 28)][lat_long[j]] + climatology_dict[(3, 1)][lat_long[j]])

                # Otherwise, get climatology value for the month and day
                else:
                    day_climatology = climatology_dict[(int(date[i, 1]), int(date[i, 2]))][lat_long[j]]

                # Add to the dictionary
                temp_grid[key][lat_long[j]] = day_climatology + day_temperatures[j]
            
    return temp_grid

# temp_dict will contain temperature grid values from 1990 to present day (mid-2018)
temp_dict = {}

# nc files containing the temperature grid values from 1990 to present day
daily_grid_files = ['Complete_TAVG_Daily_LatLong1_1990.nc', 'Complete_TAVG_Daily_LatLong1_2000.nc',
                    'Complete_TAVG_Daily_LatLong1_2010.nc']

for grid_file in daily_grid_files:
    decade_dict = load_temperature_grid('daily_land/' + grid_file)
    
    temp_dict = {**temp_dict, **decade_dict}
    
# Save dictionary into a file
with open('temperature_us_grid.pickle', 'wb') as fp:
    pickle.dump(temp_dict, fp, protocol = pickle.HIGHEST_PROTOCOL)
