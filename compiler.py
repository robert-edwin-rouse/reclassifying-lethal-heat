"""
Configuration for the data paths and random forest model to reproduce the
results in the Reclassifying Lethal Heat paper along with a list of features
and formatted plotting labels.
"""

import config
import math
import statistics
import difflib
import cdsapi
import numpy as np
import pandas as pd
import datetime as dt
import geopy as ge
import xarray as xr
import country_converter as coco


lhw_data = config.lhw_data
bmi_data = config.bmi_data
unp_data = config.unp_data
ghx_data = config.ghx_data
output = config.lethal_heat_data


def geolocate(latitude, longitude):
    """
    Returns the closest city (or heavily populated place) and containing 
    country for given coordinates.

    Parameters
    ----------
    latitude : float
        Latitude of location.
    longitude : float
        Longitude of location.

    Returns
    -------
    city : string
        Containing or closest city.
    country : string
        Containing country.
    """
    geolocator = ge.geocoders.Nominatim(user_agent=config.email)
    location = geolocator.reverse(f"{latitude},{longitude}",
                                  addressdetails=True, language='en')
    address = location.raw['address']
    city = (address.get('city') or address.get('town') or
            address.get('state') or address.get('county') or '')
    country = address.get('country', '')
    return city, country


def grid_square(lat, lon):
    """
    Rounds up latitude and longitude point data to a 1° x 1° containing grid cell

    Parameters
    ----------
    lat : float
        Latitude of location.
    lon : float
        Longitude of location.

    Returns
    -------
    grid_square_actual : list
        List of containing grid square corner coordinates in the following
        order: maximum latitude, minimum longitude, minimum latitude,
        maximum longitude.
    """
    latmax = max(math.ceil(lat), math.floor(lat))
    latmin = min(math.ceil(lat), math.floor(lat))
    lonmax = max(math.ceil(lon), math.floor(lon))
    lonmin = min(math.ceil(lon), math.floor(lon))
    grid_square_actual = [latmax, lonmin, latmin, lonmax]
    return grid_square_actual


def fuzzy_match(x, scan_array, threshold=0.9):
    """
    Matches 

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    scan_array : TYPE
        DESCRIPTION.
    threshold : TYPE, optional
        DESCRIPTION. The default is 0.9.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    aliases = difflib.get_close_matches(
        x, scan_array, len(scan_array), threshold)
    if not aliases:
        return x
    else:
        closest = statistics.mode(aliases)
        return closest


def mixed_daily_resample(array):
    """
    Resamples hourly xarray data to daily maximum temperature and mean
    windspeed and humidity.

    Parameters
    ----------
    array : xarray dataset
        Hourly meteorological data.

    Returns
    -------
    xarray dataset
        Resampled dataset with daily statistics.
    """
    return xr.Dataset({
        "temperature": array["t"].resample(time="24h").max(),
        "windspeed": ((array["u"]**2 + array["v"]**2)**0.5).resample(time="24h").mean(),
        "humidity": array["r"].resample(time="24h").mean(),
    })


def time_boundaries(heatwave_start, window_lengths):
    """
    Sets upper and lower temporal bounds for slicing meteorology
    Parameters
    ----------
    heatwave_start : datetime.date
        DESCRIPTION.
    window_lengths : TYPE
        DESCRIPTION.

    Returns
    -------
    lower_bounds : TYPE
        DESCRIPTION.
    upper_bound : TYPE
        DESCRIPTION.
        
    """
    lower_bounds = [heatwave_start -
                    dt.timedelta(days=x) for x in window_lengths]
    upper_bound = heatwave_start - dt.timedelta(days=1)
    return lower_bounds, upper_bound


def antecedent_adaptation(array, hw_start, years=10):
    """
    

    Parameters
    ----------
    array : TYPE
        DESCRIPTION.
    hw_start : TYPE
        DESCRIPTION.
    years : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    average : TYPE
        DESCRIPTION.

    """
    cache = []
    for y in range(years):
        past_end = hw_start - dt.timedelta(days=(365*y+1))
        past_start = past_end - dt.timedelta(days=30)
        t = np.mean(array.sel(time=slice(past_start, past_end))).values.item()
        cache.append(t)
    average = np.mean(cache)
    return average


def ERA5_single_retrieval(country, city, grid_square, year, pressure_level=1000):
    """
    

    Parameters
    ----------
    country : TYPE
        DESCRIPTION.
    city : TYPE
        DESCRIPTION.
    grid_square : TYPE
        DESCRIPTION.
    year : TYPE
        DESCRIPTION.
    pressure_level : TYPE, optional
        DESCRIPTION. The default is 1000.

    Returns
    -------
    None.
    """
    client = cdsapi.Client()
    era5_file = str(country) + "_" + str(city) + "_" + \
        str(year) + "_" + "era5heat.grib"
    client.retrieve("reanalysis-era5-pressure-levels",
                    {"product_type": ["reanalysis"],
                     "area": grid_square,
                     "variable": ["relative_humidity", "temperature",
                                  "u_component_of_wind", "v_component_of_wind"],
                     "pressure_level": [str(pressure_level)],
                     "data_format": "grib",
                     "download_format": "unarchived",
                     "year": [str(year)],
                     "month": ["01", "02", "03", "04", "05", "06",
                               "07", "08", "09", "10", "11", "12"],
                     "day": ["01", "02", "03", "04", "05", "06", "07", "08",
                             "09", "10", "11", "12", "13", "14", "15", "16",
                             "17", "18", "19", "20", "21", "22", "23", "24",
                             "25", "26", "27", "28", "29", "30", "31"],
                     "time": ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00",
                              "06:00", "07:00", "08:00", "09:00", "10:00", "11:00",
                              "12:00", "13:00", "14:00", "15:00", "16:00", "17:00",
                              "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"],
                     }, era5_file).download()


def ERA5_combine(country, city, grid_square, year_range, pressure_level=1000):
    """
    

    Parameters
    ----------
    country : TYPE
        DESCRIPTION.
    city : TYPE
        DESCRIPTION.
    grid_square : TYPE
        DESCRIPTION.
    year_range : TYPE
        DESCRIPTION.
    pressure_level : TYPE, optional
        DESCRIPTION. The default is 1000.

    Returns
    -------
    None.
    """
    for year in year_range:
        era5_file = str(country) + "_" + str(city) + "_" + \
            str(year) + "_" + "era5heat.grib"
        try:
            array = xr.open_dataset(era5_file)
            print(array.info)
        except:
            ERA5_single_retrieval(
                country, city, grid_square, year, pressure_level)
            print("Now downloading data for: " + str(country) + "_" +
                  str(city) + "_" + str(year))
    era5_file_glob = str(country) + "_" + str(city) + "_*_" + "era5heat.grib"
    era5_file_out = str(country) + "_" + str(city) + "era5heat.nc"
    era5_data = xr.open_mfdataset(era5_file_glob, concat_dim="time")
    era5_data = mixed_daily_resample(era5_data)
    era5_data.to_netcdf(path=era5_file_out)


def ERA_download(country, city, grid_square, year_range, pressure_level=1000):
    """
    

    Parameters
    ----------
    country : TYPE
        DESCRIPTION.
    city : TYPE
        DESCRIPTION.
    grid_square : TYPE
        DESCRIPTION.
    year_range : TYPE
        DESCRIPTION.
    pressure_level : TYPE, optional
        DESCRIPTION. The default is 1000.

    Returns
    -------
    None.

    """
    era5_file = str(country) + "_" + str(city) + "era5heat.nc"
    try:
        array = xr.open_dataset(era5_file)
    except:
        ERA5_combine(country, city, grid_square, year_range, pressure_level)
        array = xr.open_dataset(era5_file)
    print("Combined file downloaded: " + str(array is not None))


def heatwave_parameter_extraction(country, city, lat, lon, start, end, windows):
    array = xr.open_dataset(str(country) + "_" + str(city) + "era5heat.nc")
    array = array.interp(coords={'longitude': lon, 'latitude': lat},
                         method='nearest')

    bounds = time_boundaries(start, windows)
    max_t = np.max(array.sel(time=slice(start, end)).temperature).values.item()
    mean_w = np.mean(array.sel(time=slice(start, end)).windspeed).values.item()
    mean_h = np.mean(array.sel(time=slice(start, end)).humidity).values.item()
    ante_30t = np.mean(array.sel(time=slice(bounds[0][0],
                                            bounds[1])).temperature).values.item()
    ante_90t = np.mean(array.sel(time=slice(bounds[0][1],
                                            bounds[1])).temperature).values.item()
    ante_180t = np.mean(array.sel(time=slice(bounds[0][2],
                                             bounds[1])).temperature).values.item()
    ante_30h = np.mean(array.sel(time=slice(bounds[0][0],
                                            bounds[1])).humidity).values.item()
    ante_90h = np.mean(array.sel(time=slice(bounds[0][1],
                                            bounds[1])).humidity).values.item()
    ante_180h = np.mean(array.sel(time=slice(bounds[0][2],
                                             bounds[1])).humidity).values.item()
    adapt_t = antecedent_adaptation(array.temperature, start, years=10)
    adapt_h = antecedent_adaptation(array.humidity, start, years=10)
    return max_t, mean_w, mean_h, ante_30t, ante_90t, ante_180t, ante_30h, ante_90h, ante_180h, adapt_t, adapt_h


def remove_spaces(s):
    """
    

    Parameters
    ----------
    s : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if isinstance(s, str):
        return s.strip().replace(' ', '')
    return s


def best_fit_slope(xs, ys):
    """
    

    Parameters
    ----------
    xs : TYPE
        DESCRIPTION.
    ys : TYPE
        DESCRIPTION.

    Returns
    -------
    m : TYPE
        DESCRIPTION.

    """
    m = (((np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)) /
         ((np.mean(xs)*np.mean(xs)) - np.mean(xs*xs)))
    return m


def avg(xs, ys):
    """
    

    Parameters
    ----------
    xs : TYPE
        DESCRIPTION.
    ys : TYPE
        DESCRIPTION.

    Returns
    -------
    a : TYPE
        DESCRIPTION.

    """
    a = np.sum(xs*ys)/np.sum(xs)
    return a


"""
Load in lethal heatwave source data and geoprocess data, identifying the nearest
population centre, or other, and preparing the core dataframe for the appending
or additional datasets.
"""
df = pd.read_csv(lhw_data)
df['Coordinates'] = df['CodeSite'].str.split("_")
df[['Longitude', 'Latitude']] = pd.DataFrame(df.Coordinates.tolist(),
                                             index=df.index)
df['Latitude'] = df['Latitude'].astype(float)
df['Longitude'] = df['Longitude'].astype(float)
coordinates = df[['Longitude', 'Latitude']].copy()
coordinates = coordinates.drop_duplicates()
coordinates = coordinates.reset_index(drop=True)
coordinates['City'], coordinates['Country'] = zip(*coordinates.apply(lambda x: geolocate(x['Latitude'], x['Longitude']),
                                                                     axis=1))
coordinates['Grid_Square'] = coordinates.apply(
    lambda x: grid_square(x['Latitude'], x['Longitude']), axis=1)
coordinates.apply(lambda x: ERA_download(x['Country'],
                                         x['City'],
                                         x['Grid_Square'],
                                         [x for x in range(1970, 2015)],
                                         pressure_level=1000), axis=1)
df = pd.merge(df, coordinates, on=['Longitude', 'Latitude'])
columns_retained = ['StartDate', 'EndDate', 'DocuMortHetWav', 'Latitude',
                    'Longitude', 'Grid_Square', 'City', 'Country']
df = df.drop(list(set(list(df)) - set(columns_retained)), axis=1)
df['GEO'] = coco.convert(df['Country'], src="regex", to="ISO3")
df['StartDate'] = pd.to_datetime(df['StartDate'], errors='coerce').dt.date
df['EndDate'] = pd.to_datetime(df['EndDate'], errors='coerce').dt.date
df['Year'] = pd.to_datetime(df['StartDate']).dt.strftime(
    '%Y').apply(pd.to_numeric)


"""
Extract heatwave instantaneous and antecedent meteorology and add to the main
heatwave dataframe.
"""
headers = ['Max_Temperature', 'Mean_Humidity', 'Mean_Windspeed',
           'Delta_Temp_30', 'Delta_Temp_90', 'Delta_Temp_180',
           'Delta_Humid_30', 'Delta_Humid_90', 'Delta_Humid_180',
           'Adaptive_Temperature', 'Adaptive_Humidity',]
df['Extracted'] = df.apply(lambda x: heatwave_parameter_extraction(x['Longitude'],
                                                                   x['Latitude'], x['Country'], x['City'],
                                                                   x['StartDate'], x['EndDate']), axis=1)
df[headers] = pd.DataFrame(df['Extracted'].tolist(), index=df.index)


"""
Load in the Lancet BMI data, combine male/female BMI data, match the location
data format to the core dataframe, and then merge with the core dataframe.
"""
bmi_df = pd.read_csv(bmi_data, encoding="ISO-8859-1")
bmi_df['Mean BMI'] = bmi_df['Mean BMI'].apply(pd.to_numeric)
male = bmi_df[bmi_df['Sex'] == 'Men'][[
    'Country/Region/World', 'Year', 'Mean BMI']]
female = bmi_df[bmi_df['Sex'] == 'Women'][[
    'Country/Region/World', 'Year', 'Mean BMI']]
bmi_df = pd.merge(male, female, on=['Country/Region/World', 'Year'])
bmi_df['Mean_BMI'] = bmi_df[['Mean BMI_x', 'Mean BMI_y']].mean(axis=1)
bmi_df = bmi_df.drop(['Mean BMI_x', 'Mean BMI_y'], axis=1)
bmi_df['Country/Region/World'] = bmi_df['Country/Region/World'].apply(
    lambda x: fuzzy_match(x, coordinates['Country']))
bmi_df = bmi_df.rename(columns={bmi_df.columns[0]: 'Country'})
df = pd.merge(df, bmi_df, on=['Country', 'Year'], how='left')
df['GEO'] = coco.convert(df['Country'], src="regex", to="ISO3")


"""
Load in the UN Population data, extract average age and the population gradient
for each country, match the location data format to the core dataframe, and 
then merge with the core dataframe.
"""
unp_df = pd.read_csv(unp_data)
unp_df = unp_df.rename(columns={'ISO3 Alpha-code': 'GEO'})
mid_points = [2 + 5*x for x in range(21)]
age_bands = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34',
             '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69',
             '70-74', '75-79', '80-84', '85-89', '90-94', '95-99', '100']
for band in age_bands:
    unp_df[band] = unp_df[band].apply(remove_spaces)
    unp_df[band] = pd.to_numeric(unp_df[band], errors='coerce')
unp_df = unp_df.fillna(0)
pyramid_array = unp_df[age_bands].to_numpy()

age_grads = np.apply_along_axis(
    best_fit_slope, axis=1, arr=pyramid_array, ys=mid_points)
age_means = np.apply_along_axis(avg, axis=1, arr=pyramid_array, ys=mid_points)
unp_df = unp_df.assign(Gradient=pd.Series(age_grads))
unp_df = unp_df.assign(Avg_Age=pd.Series(age_means))
columns_retained = ['GEO', 'Year', 'Age_Gradient', 'Mean_Age']
unp_df = unp_df.drop(list(set(list(unp_df)) - set(columns_retained)), axis=1)
df = pd.merge(df, unp_df, on=['GEO', 'Year'], how='left')


"""
Load in the GHX SDI data, extract mean SDI for each country and then merge 
with the core dataframe.
"""
ghx_df = pd.read_csv(ghx_data)
columns_retained = ['Year', 'GEO', 'mean_sdi']
ghx_df = ghx_df.drop(list(set(list(ghx_df)) - set(columns_retained)), axis=1)
ghx_df = ghx_df.rename(columns={"mean_sdi": "Mean_SDI"})
df = pd.merge(df, ghx_df, on=['GEO', 'Year'], how='left')


"""
Export the compiled dataset for use in training the machine learning models.
"""
df.to_csv(output)
