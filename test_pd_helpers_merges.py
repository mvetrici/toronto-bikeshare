import unittest
from folderProcessor import folderProcessor
from pd_helpers import station_merge_on_trip, get_label

TRIPS = "test-bikeshare-ridership-2023"
DATA = 'other-datasets-2023'


class Testadd_cols(unittest.TestCase):

    def test_station_merge_on_trip(self):
        base_df = folderProcessor(TRIPS).get_obj().get_df() 
        add_df = folderProcessor(DATA).get_obj(dtype='BikeStation').get_df()
        station_merged = station_merge_on_trip(base_df, add_df, remove_extra_col=True)
        print(station_merged)
        station_merged.info()
        for column in station_merged.columns:
            print(station_merged[column])

    def test_get_label(self):
        possibles = ['Trip_Id', 'Trip_Duration', 'Start_Station_Id']
        result = get_label(possibles, 'station id')
        self.assertEqual(result, None)

if __name__ == '__main__':
    unittest.main()

    # pass ClassName() as argument if multiple classes
