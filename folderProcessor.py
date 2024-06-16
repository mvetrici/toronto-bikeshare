import numpy as np
import pandas as pd
import os
from dfObj import dfObj, add_col, IncompatibleDataframes

MAX_LENGTH = 120

class folderProcessor():
    def __init__(self, folder_name: str, test: str = None):
        """Get a list of the dataframes in the folder <folder_name>.
        self.name: folder name, str
        self.dfs: list of dataframe objects, list[pd.Dataframe]
        """
        self.name = folder_name
        self._types = []
        self._dfs = []

        folder_path = os.path.abspath(folder_name)
        folder_paths = [file for file in os.listdir(folder_path)]
        
        if test == 'test':
            folder_paths = [file for file in os.listdir(folder_path) if file.endswith('08.csv')] # or file.endswith('09.csv')]

        for file in folder_paths:
            print(f"File '{file}' is being processed.")
            data_path = os.path.join(folder_path, file)
            df = df_from_file(data_path) # clean the data into a dataframe
            df_obj = make_df(data_path, df) # pass dataframe to make a dfObj
            self._dfs.append(df_obj)
            self._types.append(df_obj.get_type())
        return 
    
    # Pure
    def get_ods(self, add_folder: 'folderProcessor') -> list[dfObj]:
        # TODO! make faster; pass file name to function instead of folderProcessor
        # then use same logic as 'test'
        ret = []
        for base_obj in self._dfs: # base_obj is the base merging dataframe
            # only need station data from add_folder
            add_obj = add_folder._dfs[add_folder._types.index('BikeStation')]
            # type compatibility is handled by dfObj.basic_merge()
            od_obj = base_obj.od_merge(add_obj)
            # rename with columns at one point?? 
                # EDIT THIS once it's clear what the groupby output is
                # df1.rename({"August":'August origins'}, axis=1, inplace=True)
                # df2.rename({"August":'August destinations'}, axis=1, inplace=True)
            ret.append(od_obj)
        return ret
    
    def merge_all(self, add_folder: 'folderProcessor') -> list[dfObj]:
        ret = []
        for base_obj in self._dfs: # base_obj is the base merging dataframe
           for add_obj in add_folder._dfs:
                try: # type compatibility is handled by dfObj.basic_merge()???
                    new_obj = base_obj.basic_merge(add_obj)
                    ret.append(new_obj)
                except IncompatibleDataframes:
                    pass
        return ret
    
    # Info
    def __str__(self) -> str:
        ret = 'Folder contents of ' + self.name + ':'
        for df_obj in self._dfs:
            ret += str(df_obj)
        return ret
    
    def getinfo(self):
        for df_obj in self._dfs:
            print(df_obj.name)
            df_obj.getinfo()
        return

    # Mutate
    def add_col(self, names: list[str]):
        for df_obj in self._dfs:
            new_df = add_col(df_obj, names)
            df_obj.set_df(new_df)
        return 
    
    # Mutate
    def concat_folder(self) -> pd.DataFrame:
        if len(self._dfs) == 0:
            print("No dataframes to combine")
            return
        if self._types != [self._dfs[0].get_type() for i in range(len(self._types))]:
            print("Exception: Incompatible data types")
            return
        output = self._dfs[0].get_df()
        for df_obj in self._dfs[1:]:
            output = pd.concat([output, df_obj.get_df()], axis=0, ignore_index=False)  
            # try ignore index?
            # reset index?
        return output
    
# // end class definition

# pure helper
def make_df(path: str, df: pd.DataFrame, dtype: str = None) -> dfObj:
    name = os.path.basename(path)
    if "bikeshare" in path:
        dtype = "Trip"
    if 'weather' in os.path.basename(path):
        dtype = 'Weather'
    if 'station_id' in df.columns:
        dtype = "BikeStation"
    if 'stop_id' in df.columns:
        dtype = "TTCStation"
    return dfObj(name, df, dtype)
    
# global helper
def df_from_file(path: str, encoding: str = 'cp1252') -> pd.DataFrame:
    """<path> is a path to a csv file
    <dtype> can be Trip, Weather, BikeStation, TTCStation
    Clean dataframe <self> (remove columns and remove NA rows) and remove BOM
    """
    path = os.path.abspath(path)
    df = pd.read_csv(path, encoding=encoding)
    print("DataFrame created")
    print("Original length:", len(df))
        
    # reformat column titles
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace('"', '')
    df.columns = df.columns.str.replace('  ', ' ')
    df.columns = df.columns.str.replace(' ', '_')

    # remove BOM and columns with mostly NAs
    col_drop = ['rental_uris', "obcn", "short_name", 'nearby_distance', '_ride_code_support']
    for col in df.columns:
        col1 = col.encode('cp1252').decode('utf-8-sig', 'ignore')
        df.rename(columns={col: col1}, inplace=True)
        if col1 in col_drop:
            if int(df[col1].isna().sum()) > 0.5*len(df) or col1 in col_drop:
                df.drop(col1, axis=1, inplace=True)
    
    # remove rows with NAs in columns in row_drop
    row_drop = ["End_Station_Id", 'Min_Temp_(°C)', 'Mean_Temp_(°C)', 'Max_Temp_(°C)']
    for col in row_drop:
        if col in df.columns:
            df.dropna(subset=[col], axis=0, how='any', inplace=True)
    # self.length = len(self.df)
    print("Cleaned length:", len(df))

    if "Trip_Duration" in df.columns:
        df["Trip_Duration_(min)"] = round(df["Trip_Duration"]/60, 2)
        df = df.loc[(df["Trip_Duration_(min)"] <= MAX_LENGTH) 
                    & (df["Trip_Duration_(min)"] >= 2)]
    if 'Start_Station_Id' in df.columns:
        df = df.loc[df['Start_Station_Id'] != df['End_Station_Id']]
        print("Filtered length:", len(df))
    df.reset_index(inplace=True, drop=True)
    return df

# tester = np.arange(len(df))
# if (df['index'] == tester).all():
#     df.drop('index', axis=1, inplace=True)