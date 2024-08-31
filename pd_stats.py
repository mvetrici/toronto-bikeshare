from pd_helpers import add_col_Date
import pandas as pd
import re
from gpd_helpers import run_map 
from pd_helpers import get_count_table, InvalidColError
from typing import Any

def top_n_stationpairs(df: pd.DataFrame, n: int, filter_day: Any = False, new_col_name: str = 'count') -> pd.DataFrame:
    """
    Returns dataframe with n most common station-pairs (start and end id) and their trip count 
    *df* must be trip data (optionally with station information like lat/lon)
    if n > number of trips in the dataset, all trips are counted in the trip count.
    *filter_day* specifies a specific day in the trip dataset
    Does not drop na values
    """
    bycol = ['start station id', 'end station id']
    if filter_day:
        df = filter_by_day(df, filter_day)
    trip_counts = get_count_table(df, new_col_name=new_col_name, bycol=bycol)
    sorted_trips = trip_counts.sort_values(new_col_name, ascending=False)
    top_trips = sorted_trips.drop_duplicates().iloc[0:n].reset_index(drop=True) 
    # dropna() optionally
    return top_trips

def top_n_stations(df: pd.DataFrame, n: int, filter_day: Any = False, new_col_name: str = 'count', o_or_d: str = 'start') -> pd.DataFrame:
    """
    Returns dataframe with n most common stations and their trip count
    (either origin or destination stations based on *o_or_d*)
    *df* must be trip data (optionally with station information like lat/lon)
    if n > number of trips in the dataset, all trips are counted in the trip count.
    """
    if re.search(r'start', o_or_d, flags=re.I) or re.search(r'origin', o_or_d, flags=re.I):
        bycol = ['start station id']
    elif re.search(r'end', o_or_d, flags=re.I) or re.search(r'dest', o_or_d, flags=re.I):
        bycol = ['end station id']
    else:
        raise InvalidColError(o_or_d)
    
    if filter_day:
        df = filter_by_day(df, filter_day)

    trip_counts = get_count_table(df, new_col_name=new_col_name, bycol=bycol)
    sorted_trips = trip_counts.sort_values(new_col_name, ascending=False)
    top_trips = sorted_trips.drop_duplicates().iloc[0:n].reset_index(drop=True) 
    # dropna() optionally; iloc[0:n] works even if n > len(dataframe)
    return top_trips

# helper
def filter_by_day(df: pd.DataFrame, day: Any) -> pd.DataFrame:
    """Filters df to include observations on day (mutates)"""
    add_col_Date(df)
    return df.loc[df['date'] == day]

def busiest_day(df: pd.DataFrame):
    """Returns date object of busiest day by trip count (mutates)"""
    add_col_Date(df)
    date_col = 'date'
    trips_per_day = get_count_table(df, bycol=[date_col])
    sorted_trips = trips_per_day.sort_values('count', ascending=False)
    return sorted_trips[date_col].iloc[0]

def run_nodes(df: pd.DataFrame, top_n: int = 20):
    """Converts *df* to station and plots them;
    *df* is trips with merged geo data."""
    nodes = top_n_stations(df, n=top_n)
    run_map(nodes) 
    return 


