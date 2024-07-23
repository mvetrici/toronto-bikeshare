from folderProcessor import folderProcessor
from pd_helpers import df_from_file, get_col_count, get_label, add_col_Periods
import pandas as pd
import os
from gpd_helpers import run_map 

TRIPS = "bikeshare-ridership-2023"
DATA = "other-datasets-2023"

def check_write():
    checker = input("Are you SURE you want to write this file? (y/n) ")
    if checker in ['yes', 'y', 'YES', 'Yes', 'Y']:
        return True
    return False

def write_path(title) -> str:
    """Returns file path so that saved file goes to the directory
    that holds the repo rather than in the repo."""
    return os.path.join(os.path.dirname(os.getcwd()), title)

def DEAD_day_count(df: pd.DataFrame):
    """DEPRECATED"""
    # df['day_count'] = df.groupby(['date', 'Start_Station_Id'])['Start_Station_Id'].transform('count')

    # df = add_col(df, ['timeperiod'])
    # df = pd.get_dummies(df, columns=["timeperiod_start"], drop_first=True)
    # print(df)
    # dfObj('test-with-groupby', df).write()
    pass

def top_n_stations(df: pd.DataFrame, n: int, write: bool = False) -> pd.DataFrame | None:
    """df must be trip data with added station location
    returns n most common station pairs"""
    duration_counts = get_col_count(df, new_col_name='count', bycol=['start station id', 'end station id'], new=False, keep=['lat orig', 'lon orig', 'lat dest', 'lon dest'])
    check = duration_counts.sort_values('count', ascending=False)
    check = check.drop_duplicates().dropna().iloc[0:n]
    if write:
        file = os.path.join(os.path.dirname(os.getcwd()), f"top_{n}_trips.csv")
        check.to_csv(file)
    # o = list(check['Start_Station_Id'].iloc[0:4])
    # d = list(check['End_Station_Id'].iloc[0:4])
    else:
        return check

def top_n_stations_dur(df: pd.DataFrame, n: int = 20, write: bool = False) -> pd.DataFrame:
    """df must be trip data with added station location
    returns n longest station pairs out of N most common ones"""
    stationpair_counts = get_col_count(df, new_col_name='count', bycol=['start station id', 'end station id'], new=False)
    # stationpair_counts['mean_dur'] = stationpair_counts.groupby(['Start_Station_Id', "End_Station_Id"])[get_label(stationpair_counts.columns, 'trip duration min')].mean()
    trip_dur_min = get_label(df.columns, 'trip duration min')
    stationpair_counts['mean_dur'] = df.groupby(['Start_Station_Id', 'End_Station_Id'], dropna=False, observed=True)[trip_dur_min].transform('mean')
    ret = stationpair_counts.dropna()
    ret = ret.loc[ret['mean_dur'] > df[trip_dur_min].mean()]
    ret = ret.sort_values('count', ascending=False)
    ret = ret.filter(['Start_Station_Id', "End_Station_Id", 'lat_orig', 'lon_orig', 'lat_dest', 'lon_dest', 'count', 'mean_dur'])
    ret = ret.drop_duplicates()
    if write and check_write():
        file = write_path(f"top_{n}_trips_by_duration.csv")
        ret.iloc[0:n].to_csv(file)
    # o = list(check['Start_Station_Id'].iloc[0:4])
    # d = list(check['End_Station_Id'].iloc[0:4])
    return ret

def run_nodes(df: pd.DataFrame, top_n: int = 20):
    """Converts *df* to station and plots them;
    *df* is trips with merged geo data."""
    nodes = top_n_stations_dur(df, n=top_n, write=False)
    run_map(nodes) 
    return 

def prepare_nodes(trip_data: str, add_data: str, month: str = '08', subset: bool = True) -> pd.DataFrame:
    """Merges trips data *trip_data* and station data *add_data*.
    Optionally, specify the month that is processed (August if not)."""
    bike_data = folderProcessor(trip_data, month)
    data = folderProcessor(folder_name=add_data)
    df = bike_data.combine_merge(data, station_only=True)[0].get_df()
    add_col_Periods(df, ['timeperiod'])
    if subset:
        df = df.loc[df['timeperiod'].isin(['AM', 'PM', 'Midday'])]
        df = df.sample(10000, axis=0)
    if check_write():
        df.to_csv(write_path(f"{month}-trips-station_xy.csv"), index=False)
    print(df)
    df.info()
    print(df['timeperiod'].unique())
    return df
    
def controlled_loop():
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    for month in months:
        trip_data = folderProcessor(TRIPS, month).get_obj().get_df()    
    pass 

if __name__ == '__main__':
    df = prepare_nodes(TRIPS, DATA, '11')
    
    
        

    
    # Aug top: 7059-7033
    # Aug 2nd top: 7344-7354
    




#  obj.getinfo(na=True)
# variables = ['trip_count', 'capacity_orig', 'user type', 'max temp c']
# model = makeModel(obj.get_df(), predict=variables[0], variables=variables[1:])
# model.make_lm('date')

 # makeModel(df=obj.get_df(), predict='count_orig', vars = ['capacity', 'count_dest']).make_dot(x='capacity', y='dest count')
# OD_TEST
# testdf = df_from_file(OD_TEST)
# makeModel(df=testdf, predict='count_orig', vars = ['capacity', 'count_dest']).make_dot(x='capacity', y='dest count')
