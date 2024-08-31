from file_interactors import get_trip_file, get_file_from_folder, check_write
from pd_helpers import station_merge_on_trip, station_merge_on_id, \
    station_merge_on_station_for_od
from gpd_helpers import run_map
from pd_stats import top_n_stationpairs, top_n_stations, busiest_day
from bike_trips_graphs import duration_distribution
import matplotlib.pyplot as plt
MONTHS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

def get_trip_station(trip_folder: str, data_folder: str, month: str = '08', od: bool = False):
    """Merges trip and station data. Finds od if od == True"""
    trips = get_trip_file(folder_name=trip_folder, month=month)
    stations = get_file_from_folder(data_folder, file_type='BikeStation')
    if od:
        ods = station_merge_on_station_for_od(trips, stations)
        check_write(ods)
        return ods 
    return station_merge_on_trip(trips, stations)

def gpd_map(trip_folder: str, data_folder: str, month: str = '01'):
    """Maps trips from trip_folder with station locations in data_folder"""
    trip_station = get_trip_station(trip_folder, data_folder, month)
    run_map(trip_station)

# Wrappers for pd_stats.py
def duration_distribution_wrapper(trip_folder: str, data_folder: str, month: str):
    month = '01'
    df = get_trip_station(trip_folder, data_folder, month)
    top_n = top_n_stationpairs(df, 10, False)
    print(top_n)
    duration_distribution(df, top_n, title=month, group_by_period=False)

def top_n_stationpairs_wrapper(trip_folder: str, n: int = 10):
    trips = get_trip_file(folder_name=trip_folder, month='01')
    prev = top_n_stationpairs(trips, n, False, 'trip count (month ' + '01' + ')')
    prev = prev.groupby(['Start_Station_Id', 'End_Station_Id'], observed=False).sum()
    print(prev)
    for month in MONTHS[1:]:
        pass
        # trip_station = get_trip_station(TRIPS, DATA, month)
        trips = get_trip_file(folder_name=trip_folder, month=month)
        top_n = top_n_stationpairs(trips, n, False, 'trip count (month ' + month + ')')
        top_n = top_n.groupby(['Start_Station_Id', 'End_Station_Id'], observed=False).sum()
        prev = prev.merge(top_n, left_index=True, right_on=['Start_Station_Id', 'End_Station_Id'], how='outer').fillna(0)
        print(prev)
    
    check_write(prev, f"top_{n}_trips_all_months.csv")
    if n <= 100:
        prev.plot(kind='bar')
        plt.show()


def top_n_stations_wrapper(trip_folder: str, data_folder: str, n: int = 10):
    """Outputs csv file with top n stations based on trip origin count
      (or all stations if n > total number of stations) for """
    trips = get_trip_file(folder_name=trip_folder, month='01')
    prev = top_n_stations(trips, n, o_or_d='start', new_col_name='origin trip count (month ' + '01' + ')')
    # prev = prev.groupby(['Start_Station_Id'], observed=False).sum()
    print(prev)
    for month in MONTHS[1:]:
        trips = get_trip_file(folder_name=trip_folder, month=month)
        top_n = top_n_stations(trips, n, o_or_d='start', new_col_name='origin trip count (month ' + month + ')')
        # top_n = top_n.groupby(['Start_Station_Id', 'End_Station_Id'], observed=False).sum()
        prev = prev.merge(top_n, how='outer', on='Start_Station_Id').fillna(0)
        print(prev)
    stations = get_file_from_folder(data_folder, file_type='BikeStation')
    print(stations)
    prev = station_merge_on_id(prev, stations)
    print(prev)

    check_write(prev, f"top_{n}_trips_all_months.csv")
    if n <= 100:
        prev.plot(kind='bar')
        plt.show()

def top_n_stations_per_day(trip_folder: str, data_folder: str, n: int = 10, month: str = '10'):
    """Returns flux of trips at each station (destination-origin)"""
    trips = get_trip_file(folder_name=trip_folder, month=month)
    stations = get_file_from_folder(folder_name=data_folder, file_type='BikeStation')
    busiest_date = busiest_day(trips)
    origins = top_n_stations(trips, n, filter_day=busiest_date, new_col_name='origin count', o_or_d='start')
    dests = top_n_stations(trips, n, filter_day=busiest_date, new_col_name='destination count', o_or_d='dest')
    top_n = station_merge_on_id(origins, dests)
    top_n['flux'] = top_n['destination count'] - top_n['origin count']
    top_n = station_merge_on_id(top_n, stations)
    check_write(top_n, f"top_{n}_stations_on_{busiest_date}")
    print(top_n)
    return top_n