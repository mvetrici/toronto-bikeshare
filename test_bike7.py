import unittest, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from bike7UGH import filter_trips, counts_by_col, make_bar, folder_concat, read_manipulate, \
    filter_weather, clean_data, make_hist, folder_processor, get_stats, merge_df

AUG_PATH = r"C:\Users\mvetr\OneDrive - University of Toronto\Desktop\Summer research\bikeshare-ridership-2023\Bike share ridership 2023-08.csv"
SEP_PATH = r"C:\Users\mvetr\OneDrive - University of Toronto\Desktop\Summer research\bikeshare-ridership-2023\Bike share ridership 2023-09.csv"
WEATHER_PATH = r"C:\Users\mvetr\OneDrive - University of Toronto\Desktop\Summer research\to-city-cen-weather-2023\weather2023.csv"

class TestTrips(unittest.TestCase):

    def test_counts_by_col(self):
        # my_data = counts_by_col(df, "Time Period")
        # my_data = folder_processor("bikeshare-ridership-2023", test='test') 
        # print(my_data.info())
        # print(my_data)
        # make_hist(my_data)
        # make_bar(my_data, "Trips by time period")
        a, b = 1, 1
        self.assertEqual(a, b)
    
    def test_get_stats(self):
        # print(get_stats(new_df, "Trip Duration (min)", grouping = "Start Station Id", col_interest=["Trip Duration (min)"]))
        my_data = {'Sunday': 733729, 'Friday': 815614, 'Monday': 738894, 'Tuesday': 833528, 'Wednesday': 842963, 'Saturday': 795266, 'Thursday': 847800}
        # make_bar(my_data, 'Days of the Week')
        month_data = {'January': 176597, 'April': 373056, 'July': 722107, 'February': 169582, 'October': 590071, 'December': 252374, 'August': 745564, 'May': 577842, 'March': 220796, 'November': 387718, 'September': 740335, 'June': 651752}

    def test_merge(self):
        """Tests trips data and MERGING"""
        # df1 = filter_trips(clean_data(AUG_PATH))
        # df = folder_concat("bikeshare-ridership-2023")
        # df2 = filter_weather(clean_data(WEATHER_PATH))
        # to_bar = counts_by_col(df, "Temperature Range")
        # print(len(df))
        # make_bar(to_bar, "Weather (aug)")
        return
    
    def test_filter_trips(self):
        """Test how trips are filtered."""
        df = read_manipulate(AUG_PATH)
        len1 = len(df)
        df = clean_data(AUG_PATH)
        len2 = len(df)
        df = filter_trips(df)
        len3 = len(df)
        print(f"Original length: {len1}, cleaned: {len2}, filtered: {len3}")
        df = df.loc[df['Start Station Id'] != df['End Station Id']]
        print(f"Original length: {len1}, cleaned: {len2}, filtered: {len3}, final: {len(df)}")



    def merge_basic(self):
        # df1 = filter_trips(clean_data(AUG_PATH))
        # df2 = filter_weather(clean_data(WEATHER_PATH))
        # df = merge_df(df1, df2, 'date') # taken care of by folder_processor(merge='merge')
        # to_bar = counts_by_col(df, "Temperature Range")
        return

class TestWeather(unittest.TestCase):

    def test_filter_weather(self):
        weather_data = filter_weather(clean_data(WEATHER_PATH))
        # self.assertEqual(a, 2)
        # print(weather_data.loc[(weather_data["Mean Temp (°C)"] < 16) & (weather_data["Mean Temp (°C)"] > 14)][["Mean Temp (°C)", 'Temperature Range']])
        # print(weather_data["Mean Temp (°C)"].describe())
        # weather_data["Mean Temp (°C)"].plot(kind='hist', bins=20)
        # df1 = filter_trips(clean_data(AUG_PATH))
        df1 = filter_trips(clean_data(SEP_PATH))

        # df1 = folder_concat('bikeshare-ridership-2023', test='test', merge='test')
        df2 = counts_by_col(df1, 'date')
        df2.reset_index(inplace=True) 
        df = merge_df(df2, weather_data, col = 'date')
        df.plot(x = 'Mean Temp (°C)', y = 'September', kind = 'scatter')
        print(df)
        # print(df)
        plt.show()

class TestStations(unittest.TestCase):

    def test_stations(self):
        stations_path = os.path.abspath("stations_data.csv")
        stations_data = clean_data(stations_path)
        df1 = filter_trips(clean_data(AUG_PATH))
        df = merge_df(df1, stations_data, col='station id')
        df1 = counts_by_col(df, "Start Station Id")
        df2 = counts_by_col(df, 'End Station Id')
        df1.rename({"August":'August origins'}, axis=1, inplace=True)
        df2.rename({"August":'August destinations'}, axis=1, inplace=True)
        print(df2)
        print(df2.columns)
        print(df2.info())
        df = merge_df(df1, df2)
        df.reset_index(inplace=True)
        df.rename({'index':'station_id'}, axis=1, inplace=True)
        df = pd.merge(df, stations_data, on='station_id')
        print(df['is_charging_station'])
        # df.to_csv('aug_od.csv', index=False)
        # df.plot(x = "is_charging_station", y='August destinations', kind='bar')
        plt.bar(x = df['is_charging_station'], height=df['August destinations'])
        plt.show()
        # print(stations_data.columns[1:])
        # rename_orig = {}
        # for item in stations_data.columns[1:]:
        #     rename_orig[item] = item + '_start'
        # print(rename_orig)
        # orig_stations = stations_data.rename(rename_orig, axis=1)
        


if __name__ == '__main__':
    unittest.main(TestTrips())

