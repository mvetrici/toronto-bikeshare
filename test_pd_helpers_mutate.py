import unittest, os
from folderProcessor import folderProcessor
import pandas as pd
from pd_helpers import get_col_count, df_from_file, \
    get_datetime_col, add_col_Date, add_col_Datetime, add_col_Periods, \
    get_count_table

TRIPS = "test-bikeshare-ridership-2023"
DATA = 'other-datasets-2023'

class Testadd_cols(unittest.TestCase):

    def test_add_col_Date(self):
        df = folderProcessor(TRIPS).get_obj().get_df()
        start_time = add_col_Date(df)
        print(df)
        self.assertEqual(start_time, "Start_Time")
    
    def test_add_col_Datetime(self):
        df = folderProcessor(TRIPS).get_obj().get_df()
        start_time = add_col_Datetime(df)
        print(df)
        self.assertEqual(start_time, "Start_Time")
    
    def test_unique(self):
        df = folderProcessor(TRIPS).get_obj().get_df()
        start_time = add_col_Datetime(df)
        new_col_name = df[start_time].dt.strftime("%B").unique()[0]
        self.assertEqual(new_col_name, 'January')
    
    def test_get_count_table(self):
        df = folderProcessor(TRIPS).get_obj().get_df()
        start_time = add_col_Date(df)
        new_col_name = df[start_time].dt.strftime("%B").unique()[0]
        prt = get_count_table(df, ['date'], new_col_name=new_col_name)
        print(prt)
        print(df['date'].unique())


if __name__ == '__main__':
    unittest.main()
