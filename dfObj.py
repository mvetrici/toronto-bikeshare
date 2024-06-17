import pandas as pd
import numpy as np

MAX_LENGTH = 120

class IncompatibleDataframes(Exception):
    print("No dataframes were added")

class NoDateColumnsError(Exception):
    print("No datetime columns were found")

class TooManyColumnsError(Exception):
    print("Too many columns were found that could be datetime objects")

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
    
    def getinfo(self):
        return self._df.info()
    
    def write(self):
        self._df.to_csv(self.name, index=False)
    
    def basic_merge(self, add_df: 'dfObj') -> 'dfObj': #, types: list[str]) -> 'dfObj': 
        """Creates a new dataframe object with merged dataframes.
        Does not mutate dataframes.
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
            for col in new_df:
                print(new_df[col])
        
        weather_type = ['Trip', 'Trip-BikeStation']
        if self._dtype in weather_type and add_df._dtype == 'Weather':
            # check for columns before merging, but only add to base
            # add to COPIED base, don't mutate class object dataframe
            new_df = merge_on(df1, df2, oncol='date')
        
        if new_df.empty:
            raise IncompatibleDataframes("Dataframes are incompatible")
        
        new_name = self.name + ' merged with ' + add_df.name
        new_dtype = self._dtype + '-' + add_df._dtype 

        # clean up new dataframe (e.g., duplicate columns)
        for col in new_df.columns: 
            if col.endswith('_y'):
                new_df.drop(col, axis=1, inplace=True)
        
        new_obj = dfObj(new_name, new_df, new_dtype)
        print(new_obj)
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
        
        new_name = self.name + 'MERGE' + add_df.name
        new_dtype = 'OD' 
        
        new_obj = dfObj(new_name, new_df, new_dtype)
        return new_obj
    
# Pure (returns new dataframe object)
def find_groupby(df: pd.DataFrame, tester: str) -> str:
    """Finds column to group data"""
    bycol = None
    if tester in df.columns: # option 1
        bycol = tester
    
    for col in df.columns: # check option 2 or 3 to look for column
        splitter = ' '
        if '_' in tester:
            splitter = '_'
        test_list = [int(item in col.lower()) for item in tester.split(splitter)]
        splitter = None
        if '_' in col:
            splitter = '_'
        if np.array(test_list).all() and len(test_list) == len(col.split(splitter)):
            bycol = col
    if bycol:
        return bycol
    else:
        print("Invalid column name")

def get_col_count(df: pd.DataFrame, bycol: str, new_col_name: str = 'count') -> pd.DataFrame:
    """Calls groupby on dataframe and returns a new dataframe with two columns
    first column is the grouper column, second is the count"""
    df = df.copy()
    bycol = find_groupby(df, bycol)
    if 'id' in bycol.lower():
        df[bycol] = df[bycol].astype(int) 
    keep = df.columns[2] # oh jeez
    for col in df.columns:
        if 'id' in col.lower():
            keep = col
    df_ret = df[[keep, bycol]].groupby(bycol, sort=False, observed=True, dropna=False).count()
    df_ret.rename({keep: new_col_name}, axis=1, inplace=True)
    df_ret[new_col_name] = df_ret[new_col_name].astype(int)
    df_ret.reset_index(inplace=True)
    return df_ret

def add_col(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Add columns after creating bins.
    Names should follow format of xxx_xxx"""
    print("Creating columns...")
    df = df.copy()
    # check if there's a datetime column or if it must be added
    datetime_cols = ['date', 'month', 'season', 'timeperiod', 'weekday']
    check_date = False
    possible = datetime_cols + ['weather']
    for name in names:
        if name in datetime_cols:
            check_date = True
        if name not in possible:
            print("Error in add_col: Column type not valid")
    if check_date:
        # look for datetime column and mutates df (the copy of the original)
        date_col = get_datetime_col(df)  # can return tuple or str
        if type(date_col) == tuple:
            date_col, date_cols = date_col[0], date_col
        print(date_col)
    if 'date' in names:
        df['date'] = df[date_col].dt.date
    if 'month' in names:
        df['month'] = df[date_col].dt.strftime("%B")
    if 'weekday' in names:
        df['weekday'] = df[date_col].dt.strftime("%A")
    if 'season' in names:
        bins = [pd.to_datetime('12/21/2022'), 
                    pd.to_datetime('03/20/2023'), 
                    pd.to_datetime('06/20/2023'), 
                    pd.to_datetime('09/20/2023'),
                    pd.to_datetime('12/20/2023'),
                    pd.to_datetime('12/31/2023')] # right inclusive, include last  
        labels = ["Winter", "Spring", "Summer", "Fall", "Winter"]
        df["season"] = pd.cut(df[date_col].dt.date, bins=bins, include_lowest=True, ordered=False, labels=labels)

    if ('timeperiod' in names) and date_cols: # checks if date_cols has been defined
        # if date_cols is defined, it's the trips dataset
        for col in date_cols:
            ext = ''
            if 'start' in col.lower():
                ext = '_start'
            if 'end' in col.lower():
                ext = '_end'
            bins = [pd.to_datetime('00:00:00').time(), pd.to_datetime('06:00:00').time(), 
                    pd.to_datetime('10:00:00').time(), pd.to_datetime('15:00:00').time(),
                    pd.to_datetime('19:00:00').time(), pd.to_datetime('23:00:00').time(),
                    pd.to_datetime('23:59:59').time()]
            labels = ["Overnight", "AM", "Midday", "PM", "Evening", "Overnight"]
            df["timeperiod" + ext] = pd.cut(df[col].dt.time, bins=bins, include_lowest=True, ordered=False, labels=labels)
    
    if 'weather' in names:  
        bins = [-16, 0, 5, 15, 30]
        labels = ["Freezing", "Cold", "Cool", "Warm"]
        for col in df.columns:
            if 'temp' in col.lower() and 'mean' in col.lower():
                col_name = col
        df["temp_ranges"] = pd.cut(df[col_name], 
                bins=bins, include_lowest=True, labels=labels)
        for col in df.columns:
            if 'precip' in col.lower() and 'mm' in col.lower() \
                and ('total' in col.lower()) and 'flag' not in col.lower():
                col_name = col
                df["precip"] = (df[col_name] > 0)
    return df

# pure, global
def station_merge(tr: pd.DataFrame, st: pd.DataFrame, od: str = None) -> pd.DataFrame:
    """Merges two dataframes based on the station id column.
    The first column in df2 (the stations data) must be station_id"""
    if not od:
        st_orig = st.rename(renamer(st.columns, '_orig', 'station_id'), axis=1) # don't rename station_id yet
        st_dest = st.rename(renamer(st.columns, '_dest', 'station_id'), axis=1)
    else: # if od is passed
        st_orig = st.copy()
        st_dest = st.copy()
    orig_id_label = find_groupby(tr, 'start station id') # returns label
    dest_id_label = find_groupby(tr, 'end station id')
    st_orig[orig_id_label] = st_orig['station_id']
    st_dest[dest_id_label] = st_dest['station_id']
    
    if od:
        orig_count_label = 'count_orig'
        dest_count_label = 'count_dest'
        orig_count = get_col_count(tr, 'start station id', orig_count_label)
        dest_count = get_col_count(tr, 'end station id', dest_count_label)
        print('check int or float')
        print(dest_count)
        
        st_orig = pd.merge(st_orig, orig_count, on=orig_id_label, how='left')
        st_dest = pd.merge(st_dest.filter(['station_id', dest_id_label]), dest_count, on=dest_id_label, how='left')

        output = pd.merge(st_orig, st_dest, on='station_id', how='outer')
        output.fillna({orig_count_label: 0, dest_count_label: 0}, inplace=True)
        output[orig_count_label] = output[orig_count_label].astype(int)
        output[dest_count_label] = output[dest_count_label].astype(int)
        # keep both axes then do output.fillna(0, inplace=True)
        output.drop([orig_id_label, dest_id_label], inplace=True, axis=1)
        return output
    # remove station_id columns since not necessary
    # TODO drop station_id???
    # output.drop('station_id', axis=1, inplace=True) # both have _x and _y
    
    output = pd.merge(tr, st_orig, on=orig_id_label, how='left')
    output = pd.merge(output, st_dest, on = dest_id_label, how='left')

    # output = pd.merge(tr, st_orig, on=name, how='left')
    # output.drop('station_id', axis=1, inplace=True)

    # st_dest[dest_id_label] = st_dest['station_id']
    # st_dest.drop('station_id', axis=1, inplace=True)
    return output

def renamer(keys, addition: str, avoider: str) -> dict:
    namer = {}
    for key in keys:
        if key != avoider: 
            namer[key] = key + addition
    return namer

def merge_on(df1: pd.DataFrame, df2: pd.DataFrame, oncol: str, how: str = 'left') -> pd.DataFrame:
    if oncol not in df1.columns:
        df1 = add_col(df1, [oncol])
    if oncol not in df2.columns:
        df2 = add_col(df2, [oncol])
    return pd.merge(df1, df2, on=oncol, how=how)

# HELPER and MUTATES
def get_datetime_col(df: pd.DataFrame) -> tuple|str:
    """Returns name of the column (or columns) that are datetime objects.
    Converts columns to datetime objects if they're not.
    Consider standardizing column names??"""
    if 'datetime' in df.columns and df['datetime'].dtype in ['datetime64[ns]', '<M8[ns]']:
        return 'datetime'
    columns = []
    for col in df.columns:
        if ('date' in col.lower() or 'time' in col.lower()) \
            and df[col].dtype == 'object':
            columns.append(col) # possible datetime columns 
    if len(columns) == 0:
        raise NoDateColumnsError
    if len(columns) > 2:
        raise TooManyColumnsError
    if len(columns) == 1:
        try:
            df[columns[0]] = pd.to_datetime(df[columns[0]])
        except ValueError:
            pass
        return columns[0]
    try:
        df[columns[0]] = pd.to_datetime(df[columns[0]])
        # new_col = "datetime" + ext
        # df.rename({col: new_col}, axis=1, inplace=True)
        return columns[0], columns[1]
    except ValueError:
        print("ValueError: Couldn't add datetime columns")