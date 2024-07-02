import geopandas as gpd
import pandas as pd
from folderProcessor import folderProcessor
from typing import Iterable 
from pd_helpers import get_label
from shapely.geometry import Point
import zipfile, os

#TODO not finished

DATA = 'other-datasets-2023'
# gdf_1.geometry.apply(lambda g: gdf_2.distance(g))

def get_coordinate(lat: Iterable[float], long: Iterable[float]):
    return gpd.points_from_xy(lat, long, crs="WGS84")

def coords_from_df(df: pd.DataFrame) -> gpd.GeoDataFrame:
    lat, long = get_label(df, 'lat'), get_label(df, 'lon')
    df = df.copy()
    df['geometry'] = gpd.GeoDataFrame(geometry=get_coordinate(df[lat], df[long]))
    return df

def load_dem_shp(zip_file):
    """Loads DEM shapefile in zipped folder.
    Assumes each zip file contains one shapefile"""
    z = zipfile.ZipFile(zip_file, 'r') # with zipfile.ZipFile(zip_file, 'r') as z:
    print(z)
    # Assuming the shapefile name inside the zip is the same as the zip file name (without extension)
    # shapefile_name = os.path.splitext(os.path.basename(zip_file))[0] + '.shp'
    gdf = gpd.read_file(z) #.extract(shapefile_name)) # read shapefile
    return gdf

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

def shortest_path(df_origin: pd.DataFrame, df_) -> pd.DataFrame:
    """Returns a dataframe with an added column that represents
    the shortest-path distance between the origin and dest 
    coordinates."""
    pass 



if __name__ == '__main__':
    # other_folder = folderProcessor(DATA)
    # df = other_folder.get_obj(dtype='BikeStation').get_df()
    zip_files = [os.path.join(DATA, f) for f in os.listdir(DATA) if f.endswith('.zip')]
    # Load all DEM shapefiles into a list of GeoDataFrames
    dems_gdf_list = [load_dem_shp(zip_file) for zip_file in zip_files]