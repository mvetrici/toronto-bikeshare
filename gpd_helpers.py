import geopandas as gpd
import pandas as pd
from typing import Iterable 
import os, re
import matplotlib.pyplot as plt
COLORS = ['yellow', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan', 'olive', 'magenta', 'lime', 'teal', 'indigo', 'maroon', 'navy', 'aquamarine', 'turquoise', 'silver', 'gold'] 
LTS = "lts_jan2024.shp"

def get_coordinate(lat: Iterable[float], long: Iterable[float]):
    return gpd.points_from_xy(long, lat, crs="WGS84")

def coords_from_df(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Converts lat and lon from *df* into coordinates (geodataframe), 
    without mutation. Uses WGS84 coordinate system."""
    for col in df.columns:
        if re.search('lat', col, re.I):
            lat = col
        if re.search('lon', col, re.I):
            long = col
    df = df.copy()
    geometry = get_coordinate(df[lat], df[long])
    return gpd.GeoDataFrame(data=df, crs="EPSG:4326", geometry=geometry) #type: ignore
    
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

def get_nodes(df: pd.DataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """df must have trips and station data with _orig and _dest coords"""
    # get node data
    origins = coords_from_df(df.loc[:,df.columns.str.endswith('_orig')])
    dests = coords_from_df(df.loc[:,df.columns.str.endswith('_dest')])
    print(origins)
    print(dests)
    return origins, dests

def run_map(trip_station: pd.DataFrame):
    """
    Plots all stations on a map along with LTS data
    trip_station must have trips and station location data 
    for origins and destinations 
    (very slow if entire month of trips is used)"""
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
