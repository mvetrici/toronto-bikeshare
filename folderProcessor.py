import pandas as pd
import os
from dfObj import dfObj
from pd_helpers import df_from_file, IncompatibleDataframes
from file_interactors import get_folder_paths, PathNotFound

# user-defined types of datasets
TYPES = ["Trip", 'Weather', "BikeStation", "TTCStation"] # or combination joined by -

class GetObjError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class NoDataframesFoundError(Exception):
    def __init__(self):
        self.message = "No dataframe objects exist"
        super().__init__(self.message)

class folderProcessor():
    def __init__(self, folder_name: str = 'Empty folder', month: str = ''):
        """Converts files in folder <folder_name> to pandas dataframes. 
        *test* is the month of interest in the format 'MM'.
        self.name: folder name (str)
        self.dfs: list of dataframe objects (list[dfObj])
        """
        # TODO: test!!
        self.name = folder_name
        self._types = []
        self._dfs = []
        keys = {}
        if folder_name != 'Empty folder':
            folder_paths = get_folder_paths(folder_name) # paths in the folder
            keys = folder_paths.keys()

        if folder_name == 'bikeshare-ridership-2023' and month:
            data_path = folder_paths.get(f'Bike share ridership 2023-{month}.csv')
            
            if data_path is not None:
                df = df_from_file(data_path)
                df_obj = make_df(data_path, df)
                self._dfs.append(df_obj)
                self._types.append(df_obj.get_type())
            else:
                raise PathNotFound(month)
        
        else: # runs for all files in folder
            for file in keys:
                print(f"File '{file}' is being processed.")
                df = df_from_file(folder_paths[file]) # clean the data into a dataframe
                df_obj = make_df(folder_paths[file], df) # pass dataframe to make a dfObj
                self._dfs.append(df_obj)
                self._types.append(df_obj.get_type())
        return 
    
    # Pure
    def get_ods(self, add_folder: 'folderProcessor') -> list[dfObj]:
        """<self> must be trip data; <add_folder> must include station data.
        Returns list of dataframes in <self> that have three columns each, 
        represeting the station id, its number of origins, and destinations"""
        # TODO!
        # then use same logic as 'test'
        ret = []
        for base_obj in self._dfs: # base_obj is the base merging dataframe
            # only need station data from add_folder
            add_obj = add_folder._dfs[add_folder._types.index('BikeStation')]
            od_obj = base_obj.od_merge(add_obj)
            # df1.rename({"August":'August origins'}, axis=1, inplace=True)
            # df2.rename({"August":'August destinations'}, axis=1, inplace=True)
            ret.append(od_obj)
        return ret
    
    # pure (in progress)
    # def multi_merge(self, add_folder: 'folderProcessor') -> list[dfObj]:
    #     """Attempts to combine each dataframe in <self> to a dataframe
    #     included in <add_folder> (e.g., add weather data, station data, etc.).
    #     Returns a list of new dfObj objects"""
    #     ret = []
    #     for base_obj in self._dfs: # base_obj is the base merging dataframe
    #        for add_obj in add_folder._dfs:
    #             try: 
    #                 # merge compatibility is handled by dfObj.basic_merge()
    #                 new_df = base_obj.basic_merge(add_obj)
    #                 ret.append(new_df)
    #             except IncompatibleDataframes:
    #                 pass
    #     return ret
    
    def combine_merge(self, add_folder: 'folderProcessor', station_only: bool = False) -> list[dfObj]:
        """Returns a list of the dataframes in self each 
        merged with all possible dataframes in add_folder.
        station_only only merges with BikeStation-type dataframes"""
        ret = []
        for base_obj in self._dfs: # base_obj is the base merging dataframe
           base = base_obj
           for add_obj in add_folder._dfs:
                try: # merge compatibility is handled by dfObj.basic_merge()???
                    if not station_only:
                        base = base.basic_merge(add_obj)
                    if station_only and add_obj.get_type() == 'BikeStation':
                        base = base.basic_merge(add_obj)
                except IncompatibleDataframes:
                    pass
        ret.append(base)
        return ret
    
    # Info
    def __str__(self) -> str:
        ret = 'Folder contents of ' + self.name + ':'
        for df_obj in self._dfs:
            ret += str(df_obj)
        return ret
    
    def get_obj(self, index: int = 0, dtype: str = '') -> dfObj:
        """index or type must be valid. Valid types:
        "Trip", 'Weather', "BikeStation", 'TTCStation'"""
        if len(self._dfs) == 0:
            raise NoDataframesFoundError()
        if index:
            if index < len(self._dfs):
                return self._dfs[index]
            else:
                raise GetObjError(f"Invalid index: folder contains {len(self._dfs)} object(s)")
        if dtype:
            if dtype not in TYPES or dtype not in self._types:
                raise GetObjError(f"Folder doesn't contain type {dtype}")
            return self._dfs[self._types.index(dtype)]
        return self._dfs[0]
    
    def get_dfs(self) -> list[dfObj]:
        return self._dfs.copy()
    
    # Mutate
    def concat_folder(self) -> pd.DataFrame:
        """Concatenates all dataframes in <self> into one."""
        if len(self._dfs) == 0:
            raise NoDataframesFoundError()
        if self._types != [self._dfs[0].get_type() for i in range(len(self._types))]:
            raise IncompatibleDataframes()
        output = self._dfs[0].get_df()
        for df_obj in self._dfs[1:]:
            output = pd.concat([output, df_obj.get_df()], axis=0, ignore_index=False)  
            # try ignore index?
            # reset index?
        return output
# // end class definition

# pure helper (in progress)
def make_df(path: str, df: pd.DataFrame, dtype: str = 'Unknown') -> dfObj:
    """Assigns types to dataframe objects based on  pathname or columns."""
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