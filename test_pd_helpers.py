import unittest
from folderProcessor import folderProcessor
import pandas as pd
from pd_helpers import get_col_count, get_folder_paths, df_from_file, \
    get_datetime_col, add_col
TRIPS = "bikeshare-ridership-2023"
JANTRIPS = r"Bike share ridership 2023-01.csv"
DATA = 'other-datasets-2023'

#TODO not finished

class TestFolder(unittest.TestCase):

    def test_get_paths(self):
        paths = get_folder_paths(TRIPS)
        b = paths.get(JANTRIPS)
        print(b)
        result = r'C:\\Users\\mvetr\\OneDrive - University of Toronto\\Desktop\\Summer research\\bike_model_06_24\\toronto-bikeshare\\bikeshare-ridership-2023\\Bike share ridership 2023-01.csv'
        pass

    def test_df_from_file(self):
        paths = get_folder_paths(TRIPS)
        df = df_from_file(paths.get(JANTRIPS))
        df.info()
        print(df["End_Station_Id"].dtype)

class AddCol(unittest.TestCase):
    
    def add_col(self):
        trips_folder = folderProcessor(TRIPS, 'test')
        other_folder = folderProcessor(DATA)
        df = trips_folder.combine_merge(other_folder).get_df()
        df = add_col(df, ['weather', 'cost'])
        df.info()
        print(df)
    
    # def test_visualize_cost(self):
    #     trips_folder = folderProcessor(TRIPS, 'test')
    #     other_folder = folderProcessor(DATA)
    #     df = trips_folder.combine_merge(other_folder).get_df()
    #     df = add_col(df, ['weather', 'cost'])

    #     print(df.groupby(['Start_Station_Id', 'End_Station_Id'])['Trip_Id'].count().sort_values(ascending=False))
    
    #     df = df.loc[lambda df: (df['Start_Station_Id'] == 7059) & (df['End_Station_Id'] == 7033)]
    #     df.plot(kind = 'hist', x=df['cost'])
    #     plt.show()
    #     print('done')
    #     # pd.plot()
    
class TestModify(unittest.TestCase):

    def test_col_count(self):
        # trips_folder = folderProcessor(TRIPS, 'test')
        # other_folder = folderProcessor(DATA)
        # df = trips_folder.combine_merge(other_folder).get_df()
        # df = add_col(df, ['weather'])
        # output = get_col_count(df, ['User_Type', 'precip'], new=True)
        # output.info()
        # print(output)
        pass
    
    def test_col_count(self):
        trips_folder = folderProcessor(TRIPS, 'test')
        other_folder = folderProcessor(DATA)
        df = trips_folder.combine_merge(other_folder).get_df()
        df = add_col(df, ['weather'])
        start = get_col_count(df, ['start station id', 'date'], new=False, keep = ['capacity orig'])
        end = get_col_count(df, ['end station id', 'date'], new=False, keep = ['capacity dest'])
        print(start)
        print(end)
        # start.to_csv('startjanbydate.csv', index=False)
        # end.to_csv('endjanbydate.csv', index=False)

class TestDateCol(unittest.TestCase):

    def test_get_datetime_col(self):
        df = df_from_file(get_folder_paths(TRIPS).get(JANTRIPS))
        get = get_datetime_col(df)
        want = ['Start_Time', 'End_Time']
        self.assertEqual(get, want)
    
    def test_get_datetime_col(self):
        trips_folder = folderProcessor(TRIPS, 'test')
        other_folder = folderProcessor(DATA)
        df = trips_folder.combine_merge(other_folder).get_df()
        get = get_datetime_col(df)
        want = ["Start_Time", "End_Time", "date", 'DateTime']
        self.assertEqual(get, want)
    
    def test_timeperiod(self):
        trips_folder = folderProcessor(TRIPS, 'test')
        other_folder = folderProcessor(DATA)
        df = trips_folder.combine_merge(other_folder).get_df()
        df = add_col(df, ['timeperiod'])
        df = pd.get_dummies(df, columns=["timeperiod"], drop_first=True)
        print(df)


if __name__ == '__main__':
    unittest.main(AddCol())
    # pass ClassName() if multiple classes

