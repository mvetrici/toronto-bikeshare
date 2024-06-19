import pandas as pd
import os
from dfObj import dfObj
from pd_helpers import df_from_file, IncompatibleDataframes, add_col, \
    get_folder_paths

class folderProcessor():
    def __init__(self, folder_name: str, test: str = None):
        """Get a list of the dataframes in the folder <folder_name>.
        self.name: folder name, str
        self.dfs: list of dataframe objects, list[pd.Dataframe]
        """
        self.name = folder_name
        self._types = []
        self._dfs = []
        folder_paths = get_folder_paths(folder_name)

        if test == 'test':
            data_path = folder_paths.get('Bike share ridership 2023-01.csv')
            df = df_from_file(data_path)
            df_obj = make_df(data_path, df)
            self._dfs.append(df_obj)
            self._types.append(df_obj.get_type())
        
        else:
            for file in folder_paths.keys():
                print(f"File '{file}' is being processed.")
                df = df_from_file(folder_paths[file]) # clean the data into a dataframe
                df_obj = make_df(folder_paths[file], df) # pass dataframe to make a dfObj
                self._dfs.append(df_obj)
                self._types.append(df_obj.get_type())
        return 
    
    # Pure
    def get_ods(self, add_folder: 'folderProcessor') -> list[dfObj]:
        # TODO!
        # then use same logic as 'test'
        ret = []
        for base_obj in self._dfs: # base_obj is the base merging dataframe
            # only need station data from add_folder
            add_obj = add_folder._dfs[add_folder._types.index('BikeStation')]
            # merge compatibility is handled by dfObj.basic_merge()
            od_obj = base_obj.od_merge(add_obj)
            # df1.rename({"August":'August origins'}, axis=1, inplace=True)
            # df2.rename({"August":'August destinations'}, axis=1, inplace=True)
            ret.append(od_obj)
        return ret
    
    def multi_merge(self, add_folder: 'folderProcessor') -> list[dfObj]:
        """Returns a list of dataframes, where each one is a different pairing"""
        ret = []
        for base_obj in self._dfs: # base_obj is the base merging dataframe
           for add_obj in add_folder._dfs:
                try: # merge compatibility is handled by dfObj.basic_merge()???
                    new_df = base_obj.basic_merge(add_obj)
                    ret.append(new_df)
                except IncompatibleDataframes:
                    pass
        return ret
    
    def combine_merge(self, add_folder: 'folderProcessor') -> dfObj:
        """Returns a dataframe merged with all possible dataframes"""
        for base_obj in self._dfs: # base_obj is the base merging dataframe
           base = base_obj
           for add_obj in add_folder._dfs:
                try: # merge compatibility is handled by dfObj.basic_merge()???
                    base = base.basic_merge(add_obj)
                except IncompatibleDataframes:
                    pass
        return base
    
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
