from folderProcessor import folderProcessor
from dfObj import dfObj
from pd_helpers import df_from_file, add_col, get_col_count, get_label
import pandas as pd
import os
from gpd_helpers import run_map

bike_path = "bikeshare-ridership-2023"
data_path = "other-datasets-2023"

def day_count(df: pd.DataFrame):
    df['day_count'] = df.groupby(['date', 'Start_Station_Id'])['Start_Station_Id'].transform('count')

    df = add_col(df, ['timeperiod'])
    df = pd.get_dummies(df, columns=["timeperiod_start"], drop_first=True)
    print(df)
    dfObj('test-with-groupby', df).write()

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

def top_n_stations_dur(df: pd.DataFrame, n: int = 20, write: bool = False):
    """df must be trip data with added station location
    returns n longest station pairs out of N most common ones"""
    stationpair_counts = get_col_count(df, new_col_name='count', bycol=['start station id', 'end station id'], new=False)
    # stationpair_counts['mean_dur'] = stationpair_counts.groupby(['Start_Station_Id', "End_Station_Id"])[get_label(stationpair_counts, 'trip duration min')].mean()
    trip_dur_min = get_label(df, 'trip duration min')
    stationpair_counts['mean_dur'] = df.groupby(['Start_Station_Id', 'End_Station_Id'], dropna=False, observed=True)[trip_dur_min].transform('mean')
    ret = stationpair_counts.dropna()
    ret = ret.loc[ret['mean_dur'] > df[trip_dur_min].mean()]
    ret = ret.sort_values('count', ascending=False)
    ret = ret.filter(['Start_Station_Id', "End_Station_Id", 'lat_orig', 'lon_orig', 'lat_dest', 'lon_dest', 'count', 'mean_dur'])
    ret = ret.drop_duplicates()
    if write:
        file = os.path.join(os.path.dirname(os.getcwd()), f"top_{n}_trips_by_duration.csv")
        ret.iloc[0:n].to_csv(file)
    # o = list(check['Start_Station_Id'].iloc[0:4])
    # d = list(check['End_Station_Id'].iloc[0:4])
    else:
        return ret


if __name__ == '__main__':
    bike_data = folderProcessor(bike_path, 'test')
    add_data = folderProcessor(data_path)
    df = bike_data.combine_merge(add_data, station_only=True)[0].get_df()
    nodes = top_n_stations_dur(df, n=20, write=True)
    # run_map(nodes)
    print(nodes)
    
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
