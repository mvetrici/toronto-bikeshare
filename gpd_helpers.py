import geopandas as gpd
import pandas as pd
from folderProcessor import folderProcessor
from typing import Iterable 
from pd_helpers import get_label, add_col, get_col_count, station_merge
from shapely.geometry import Point
import zipfile, re, os
import matplotlib.pyplot as plt
import osmnx as ox
COLORS = ['yellow', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan', 'olive', 'magenta', 'lime', 'teal', 'indigo', 'maroon', 'navy', 'aquamarine', 'turquoise', 'silver', 'gold'] 

#TODO not finished

DATA = 'other-datasets-2023'
# gdf_1.geometry.apply(lambda g: gdf_2.distance(g))

def get_coordinate(lat: Iterable[float], long: Iterable[float]):
    return gpd.points_from_xy(long, lat, crs="WGS84")

def coords_from_df(df: pd.DataFrame) -> gpd.GeoDataFrame:
    for col in df.columns:
        if 'lat' in col:
            lat = col
        if 'lon' in col:
            long = col
    # lat, long = get_label(df, 'lat'), get_label(df, 'lon')
    df = df.copy()
    # df['y'], df['x'] = df[lat], df[long]
    geometry = get_coordinate(df[lat], df[long])
    return gpd.GeoDataFrame(df, crs="EPSG:4326", geometry=geometry) # WGS84 global 

def load_dem_shp(zip_file):
    """Loads DEM shapefile in zipped folder.
    Assumes each zip file contains one shapefile"""
    # gdfs = []
    # shp_regex = "^ne_.*\.shp$"
    # with zipfile.ZipFile(zip_file, 'r') as z:
    #     print(z)
    # # Assuming the shapefile name inside the zip is the same as the zip file name (without extension)
    #     zipped_shp_namelist = z.namelist()
    #     print(zipped_shp_namelist)
    #     for filename in zipped_shp_namelist:
    #         if (re.search(shp_regex, filename)):
    #     # shapefile_name = os.path.splitext(os.path.basename(zip_file))[0] + '.shp'
    #     # gdf = gpd.read_file(z) #.extract(shapefile_name)) # read shapefile
                
    #             gdfs.append(gpd.read_file(filename))
    # print(gdfs)
    # # return gdf
    # pass

def get_elevations(folder_name: str):
    """Folder <folder_name> must contain elevation data"""
    # Example usage to load all DEM shapefiles in the "datasets" folder
    # List all zip files in the datasets folder
    zip_files = [os.path.join(folder_name, f) for f in os.listdir(folder_name) if f.endswith('.zip')]

    # Load all DEM shapefiles into a list of GeoDataFrames
    dems_gdf_list = [load_dem_shp(zip_file) for zip_file in zip_files]

    # Combine all GeoDataFrames into a single GeoDataFrame if needed
    dems_gdf = gpd.GeoDataFrame(pd.concat(dems_gdf_list, ignore_index=True), crs=dems_gdf_list[0].crs)

    # Ensure the CRS of points matches the CRS of DEMs (if different)
    points_crs = 'epsg:4326'  # WGS84 coordinate system (typical for latitude and longitude)

    # Load points from CSV using Pandas
    points_df = pd.read_csv('points.csv')

    # Convert points DataFrame to GeoDataFrame
    points_gdf = gpd.GeoDataFrame(points_df, geometry=gpd.points_from_xy(points_df.longitude, points_df.latitude), crs=points_crs)

    # Reproject points if necessary to match the CRS of DEMs
    points_gdf = points_gdf.to_crs(dems_gdf.crs)

    # Function to extract elevation for each point
    def extract_elevation(row):
        point = row.geometry
        # Find DEM that contains the point
        for idx, dem in dems_gdf.iterrows():
            if dem.geometry.contains(point):
                return dem['elevation']  # Adjust this to extract elevation attribute from DEM
        return None  # Handle cases where point is outside all DEMs

    # Apply function to each point
    points_gdf['elevation'] = points_gdf.apply(extract_elevation, axis=1)

    # Display results
    print(points_gdf[['latitude', 'longitude', 'elevation']])
    return

def load_shp(zipfile_name: str, show: bool = False) -> gpd.GeoDataFrame:
    path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), zipfile_name + '.zip')
    path = os.path.join(path, zipfile_name)
    gdf = gpd.read_file(f"zip://{path}")

    if show:
        print(gdf.crs) # EPSG:4326
        print(gdf)
        gdf['geometry'].plot()
        plt.show()
    return gdf

def shortest_path(origins: gpd.GeoDataFrame, dests: gpd.GeoDataFrame, network) -> pd.DataFrame:
    """Returns a dataframe with an added column that represents
    the shortest-path distance between the origin and dest 
    coordinates.
    <trip_st_data> data must be trip data with bike station information,
    specifically coordinates"""

    # origins, dests = origins.to_crs('EPSG:4536'), dests.to_crs('EPSG:4536')
    
    origins['x'] = origins['geometry']

    # get edge data
    if network.crs != 'EPSG:4536':
        network = network.to_crs('EPSG:4536') # TODO or maybe to this AFTER?
        print(network)
    print(origins)
    # offset for origins in EPSG:4326
    offset = 14951010 # TODO verify w/ # print(network.sindex) 
    # 100000000 too long, 10000000 too short, 40000000 too long and 20000000
    # 14951050 too wide, 14951000 too small
    # [160753, 161324]

    bbox = origins.bounds + [-offset, -offset, offset, offset]
    print(bbox)
    network_points = bbox.apply(lambda row: list(network.sindex.intersection(row)), axis=1)
    df = pd.DataFrame(network_points.reset_index())
    print(df[df.columns[1]].explode().unique())
    
    # TripBikeStation["distance"] = orig.distance(dest, align=True)
    pass

def get_nodes(df: pd.DataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """df must have trips and station data with _orig and _dest coords"""
    # get node data
    origins = coords_from_df(df.loc[:,df.columns.str.endswith('_orig')])
    dests = coords_from_df(df.loc[:,df.columns.str.endswith('_dest')])
    return origins, dests

def run_df(trips: str, other_data: str) -> pd.DataFrame:
    """<trips> represents folder with trip data
    <other_data> represents folder with location/other datasets
    Note: uses 'test'!!!"""
    
    trip_data = folderProcessor(trips, 'test').get_obj().get_df()
    # other_folder = folderProcessor(data)
    # df = trips_folder.combine_merge(other_folder)[0].get_df()
    # df = add_col(df, ['weather', 'cost', 'timeperiod'])
    duration_counts = get_col_count(trip_data, new_col_name='count', bycol=['start station id', 'end station id'], new=True) #, keep=['trip duration min'])
    check = duration_counts.sort_values('count', ascending=False)
    o = list(check['Start_Station_Id'].iloc[0:4])
    d = list(check['End_Station_Id'].iloc[0:4])
    filtered = trip_data.loc[(trip_data["Start_Station_Id"].isin(o)) & (trip_data["End_Station_Id"].isin(d))].reset_index()
    # filtered = trip_data.loc[lambda trip_data: (trip_data['Start_Station_Id'] in o) & (trip_data['End_Station_Id'] in d)]
    st = folderProcessor(other_data).get_obj(dtype="BikeStation").get_df()
    return station_merge(filtered, st)
    # return filtered

def run_map(trip_station: pd.DataFrame):
    """trip_station must have trips and station location data"""
    lts = load_shp(LTS, show=False)
    fig, ax = plt.subplots(figsize=(11, 14))
    lts.plot(ax=ax, alpha=0.5, zorder=1)
    origins, dests = get_nodes(trip_station)
    
    # for i in range(len(origins)):
    #     origins.iloc[i].plot(ax=ax, color=COLORS[i], zorder=2)
    #     dests.iloc[i].plot(ax=ax, color=COLORS[i], zorder=3)
    origins.plot(ax=ax, color="yellow", zorder=2)
    dests.plot(ax=ax, color="pink", zorder=3)
    plt.legend()
    plt.show()

TRIPS = "bikeshare-ridership-2023"
JANTRIPS = r"Bike share ridership 2023-01.csv"
DATA = 'other-datasets-2023'
LTS = "lts_jan2024.shp"

if __name__ == '__main__':
    # other_folder = folderProcessor(DATA)
    # df = other_folder.get_obj(dtype='BikeStation').get_df()
    # zip_files = [os.path.join(DATA, f) for f in os.listdir(DATA) if f.endswith('.zip')]
    #  Load all DEM shapefiles into a list of GeoDataFrames
    # dems_gdf_list = [load_dem_shp(zip_file) for zip_file in zip_files]
    trip_station = run_df(TRIPS, DATA)
    df = run_map(trip_station)
    run_map(df)
