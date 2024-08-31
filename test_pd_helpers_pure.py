import unittest
from pd_helpers import renamer, get_label, station_merge_on_trip
from pd_stats import top_n_stationpairs, busiest_day
from file_interactors import get_trip_file, get_file_from_folder
from bike_trips_graphs import duration_distribution

TRIPS = "test-bikeshare-ridership-2023"
DATA = 'other-datasets-2023'
RUN = False 

class TestPure(unittest.TestCase):
    
    def test_renamer(self):
        if RUN: 
            words = ['count', 'yes', 'no']
            ren = renamer(words, '_orig', 'no', 'noop')
            print(ren)

    def test_get_label_2(self):
        if RUN:
            possible_labels = ['HEllo', 'One_day', 'Time DAte', 'hello there']
            label = 'day one'
            result = get_label(possible_labels, label)
            print(result)
    
    def test_busiest_day(self):
        base_df = get_trip_file(TRIPS, 'mini')
        day = busiest_day(base_df)
        print(day)

class TestMain(unittest.TestCase):
    def setUp(self):
        base_df = get_trip_file(TRIPS, 'mini')
        add_df = get_file_from_folder(DATA, file_type='BikeStation')
        self.trip_station = station_merge_on_trip(base_df, add_df)

    def test_top_n_stationpairs(self):
        print(self.trip_station[['End_Station_Id', 'Start_Station_Id']])
        df = top_n_stationpairs(self.trip_station, n = 3)
        print(df)
        duration_distribution(self.trip_station, df)

if __name__ == '__main__':
    unittest.main()
