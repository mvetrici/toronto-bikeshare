import unittest
from pd_helpers import get_label, \
    station_merge_on_station_for_od, station_merge_on_id
from file_interactors import get_trip_file, get_file_from_folder
from bike_trip_interactors import get_trip_station

TRIPS = "test-bikeshare-ridership-2023"
DATA = 'other-datasets-2023'

class Testadd_cols(unittest.TestCase):

    def test_station_merge_on_trip(self):
        station_merged = get_trip_station(TRIPS, DATA, 'mini')
        print(station_merged)
        station_merged.info()
        for column in station_merged.columns:
            print(station_merged[column])

    def test_get_label(self):
        possibles = ['Trip_Id', 'Trip_Duration', 'Start_Station_Id']
        result = get_label(possibles, 'station id')
        self.assertEqual(result, None)

class TestOD(unittest.TestCase):
    def setUp(self):
        self.trips = get_trip_file(TRIPS, 'mini')
        self.stations = get_file_from_folder(DATA, file_type='BikeStation')
    
    def test_station_merge_on_station_for_od(self):
        result = station_merge_on_station_for_od(self.trips, self.stations)
        print(result['count_orig'].unique())
    
    def test_station_merge_on_id(self):
        result = station_merge_on_id(self.trips, self.stations)
        print("ON ID")
        print(result)
        

if __name__ == '__main__':
    unittest.main()
