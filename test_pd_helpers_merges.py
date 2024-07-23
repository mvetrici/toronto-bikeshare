import unittest, os
from folderProcessor import folderProcessor
import pandas as pd
from pd_helpers import station_merge_on_trip

TRIPS = "test-bikeshare-ridership-2023"
DATA = 'other-datasets-2023'


class Testadd_cols(unittest.TestCase):

    def test_station_merge_on_trip(self):
        base_df = folderProcessor(TRIPS).get_obj().get_df() 
        add_df = folderProcessor(DATA).get_obj(dtype='BikeStation').get_df()
        station_merged = station_merge_on_trip(base_df, add_df, remove_extra_col=True)
        print(station_merged)
        station_merged.info()
        for col in station_merged.columns:
            print(station_merged[col])


if __name__ == '__main__':
    unittest.main()

    # pass ClassName() as argument if multiple classes
