# hello
import pandas as pd
from pd_helpers import station_merge, add_col, merge_on, IncompatibleDataframes

MAX_LENGTH = 120


class dfObj():
    """A dataframe object. 
        name: str (name of the file)
        df: pd.DataFrame
        dtype: str (Trip, Weather, BikeStation, TTCStation, Trip-Weather, etc.)
        length: int (length of the dataframe)
        """
  
    def __init__(self, name, df: pd.DataFrame, dtype: str = 'DataFrame'):
        self.name = name
        self._df = df
        self._dtype = dtype
        self._length = len(df)
        return

    def get_df(self) -> pd.DataFrame:
        return self._df

    def set_df(self, df: pd.DataFrame): 
        self._df = df
        return

    def get_type(self) -> str:
        return self._dtype
    
    def __str__(self) -> str:
        return '\n' + f"{self.name} (type {self._dtype}):\n" + str(self.get_df())
    
    def getinfo(self, na: bool = False):
        if na:
            print(self._df.isna().sum())
        return self._df.info()
    
    def write(self):
        self._df.to_csv(self.name + '.csv', index=False)
    
    def basic_merge(self, add_df: 'dfObj') -> 'dfObj': #, types: list[str]) -> 'dfObj': 
        """Creates a new dfObj object with merged dataframes.
        Does not mutate existing dataframes.
        Function should be applied on "left" dataframe (i.e., self
        refers to a dataframe with all the valid keys)
        """
        df1 = self._df.copy()
        df2 = add_df._df.copy()
        new_df = pd.DataFrame()
        bike_type = ['Trip', 'Trip-Weather']
        if self._dtype in bike_type and add_df._dtype == 'BikeStation':
            # check for columns before merging, but only add to base
            # add to COPIED base, don't mutate class object dataframe
            new_df = station_merge(df1, df2)
            # new_df = add_col(new_df, ['trip_count'])
        
        weather_type = ['Trip', 'Trip-BikeStation']
        if self._dtype in weather_type and add_df._dtype == 'Weather':
            # check for columns before merging, but only add to base
            # add to COPIED base, don't mutate class object dataframe
            new_df = merge_on(df1, df2, oncol='date')
        
        if new_df.empty:
            raise IncompatibleDataframes("Dataframes are incompatible")
        
        new_name = f'merge-{add_df._dtype}-{self.name.split('.')[0]}'
        new_dtype = self._dtype + '-' + add_df._dtype 

        # TODO! remove duplicated columns
        for col in new_df.columns: 
            if col.endswith('_y'):
                new_df.drop(col, axis=1, inplace=True)
        
        new_obj = dfObj(new_name, new_df, new_dtype)
        return new_obj

    def od_merge(self, add_df: 'dfObj') -> 'dfObj': #, types: list[str]) -> 'dfObj': 
        """Creates a new dataframe object with merged dataframes.
        Does not mutate dataframes.
        Function should be applied on "left" dataframe (i.e., self
        refers to a dataframe with all the valid keys)
        """
        df1 = self._df.copy()
        df2 = add_df._df.copy()
        new_df = pd.DataFrame()
        bike_type = ['Trip']
        if self._dtype in bike_type and add_df._dtype == 'BikeStation':
            # check for columns before merging, but only add to base
            # add to COPIED base, don't mutate class object dataframe
            new_df = station_merge(df1, df2, 'od')
        
        if new_df.empty:
            raise IncompatibleDataframes("Dataframes are incompatible")
        
        new_name = 'OD-merge-' + self.name.split('.')[0]
        new_dtype = 'OD' 
        
        new_obj = dfObj(new_name, new_df, new_dtype)
        return new_obj