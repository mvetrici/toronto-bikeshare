import pandas as pd
import numpy as np
import os, re
from typing import Iterable

class IncompatibleDataframes(Exception):
    """Raises an exception when dataframes can't be merged"""
    def __init__(self):
        self.message = "Exception: Incompatible data types"
        super().__init__(self.message)

class NoDateColumnsError(Exception):
    def __init__(self):
        self.message = "No date columns were found"
        super().__init__(self.message)

class InvalidColError(Exception):
    def __init__(self, colname: str):
        self.message = f"<{colname}> could not be found in the data"
        super().__init__(self.message)

class NoIndexTypeColumn(Exception):
    def __init__(self):
        self.message = "No index-type column that can be converted to 'count' column"
        super().__init__(self.message)

MAX_LENGTH = 120 # time in minutes of longest possible trip
DATETYPES = ['datetime64[ns]', '<M8[ns]']


def df_from_file(path: str, encoding: str = 'cp1252') -> pd.DataFrame:
    """<path> is a path to a csv file
    <dtype> can be Trip, Weather, BikeStation, TTCStation 
    Cleans dataframe (remove columns and remove NA rows from certain columsn).
    Removes BOM.
    """
    path = os.path.abspath(path)
    df = pd.read_csv(path, encoding=encoding)
    print("DataFrame created")
    filtering, orig_length = '', len(df)
        
    # reformat column titles
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace('"', '')
    df.columns = df.columns.str.replace('  ', ' ')
    df.columns = df.columns.str.replace(' ', '_')

    # remove BOM and columns with mostly NAs
    col_drop = ['rental_uris', "obcn", "short_name", 'nearby_distance', '_ride_code_support', 'Climate_ID', 'Station_Name']
    for col in df.columns:
        col1 = col.encode('cp1252').decode('utf-8-sig', 'ignore')
        col1 = re.sub(r'\W+', '', col1)
        df.rename(columns={col: col1}, inplace=True)
        
        # remove unhelpful columns or columns with mostly NAs 
        remove = col1 in col_drop or 'flag' in col1.lower() or int(df[col1].isna().sum()) > 0.5*len(df) or 'days' in col1.lower() or 'gust' in col1.lower()
        if remove:
            df.drop(col1, axis=1, inplace=True)

        # remove rows with NAs in certain columns
        row_drop = ["End_Station_Id", 'Min_Temp_C', 'Mean_Temp_C', 'Max_Temp_C']
        dropna = col1 in row_drop
        if dropna: 
            df.dropna(subset=[col1], axis=0, how='any', inplace=True)

    shortest = len(df) # used for print statement

    try: # reformat weather data specifically
        precip_label = get_label(df.columns, 'total precip mm')
        df.drop(['Longitude_x', 'Latitude_y'], axis=1, inplace=True)
        df["precip"] = (df[precip_label] > 0)
        df['precip'] = df['precip'].astype(int)
    except InvalidColError:
        pass

    try: # reformat trip data and filter out outlier trips
        duration_label, startid_label, endid_label = get_label_list(df.columns, ['Trip_Duration', 'Start_Station_Id', 'End_Station_Id'])
        df[endid_label] = df[endid_label].astype(int)
        df["Trip_Duration_min"] = round(df[duration_label]/60, 2)
        df = df.loc[(df["Trip_Duration_min"] <= MAX_LENGTH) 
                    & (df["Trip_Duration_min"] >= 2)]
        df = df.loc[df[startid_label] != df[endid_label]]
        filtering, shortest = "(with filtering)", len(df)
    except InvalidColError:
        pass
    
    print(orig_length - shortest, "observations were removed", filtering)
    df.reset_index(inplace=True, drop=True)
    
    return df

# Pure (returns new dataframe object)
def get_label_list(possible_labels: Iterable | list[str], column_names: list[str]) -> list[str]:
    new_labels = column_names.copy()
    for i in range(len(column_names)):
        new_labels[i] = get_label(possible_labels, column_names[i])
    return new_labels

# Pure helper
def get_label(possible_labels: Iterable | list[str], label: str) -> str:
    """Finds corresponding str <label> among the options
     in <possible_labels>. Does not modify possible_labels."""
    # TODO use regex
    bycol = None
    if label in possible_labels: # option 1
        return label
    
    for col in possible_labels: # check option 2 or 3 to look for column
        splitter = '_' if '_' in label else ' '
        test_list = [int(item in col.lower()) for item in label.split(splitter)]
        
        splitter = '_' if '_' in col else ' '
        if np.array(test_list).all() and len(test_list) == len(col.split(splitter)):
            bycol = col
    if bycol:
        return bycol
    else:
        raise InvalidColError(label)

# Pure (will be improved)
def get_col_count(df: pd.DataFrame, bycol: list[str], 
                  new_col_name: str = 'count', 
                  new: bool = False, 
                  keep: list = []) -> pd.DataFrame:
    """Calls groupby on dataframe and returns a new dataframe with two columns:
    first column is the grouper column, second is the count. 
    <bycol> is a list of one or more labels
    """
    bycol= get_label_list(df.columns, bycol)
    df = df.reset_index() # add 'index' column that can be used for count
    if new: # create new dataframe
        df = df.groupby(bycol, observed=False).count().reset_index()

        # optionally add sort=False, dropna=False
        df.rename({'index': new_col_name}, axis=1, inplace=True)
        return df
        # df_out = merge_on(df_out, df.filter([bycol[0]] + keep), oncol=bycol[0], how='left').drop_duplicates().reset_index(drop=True)
    
    # ELSE: add column to existing dataframe
    df[new_col_name] = df.groupby(bycol, dropna=False, observed=True)['index'].transform('count')
    if keep: # return subset of columns
        keep = get_label_list(df.columns, keep)
        keep_list = [item for item in keep if item in df.columns]
        if keep_list != keep:
            print(f"Could not keep all columns in {keep} because they were not valid columns")
        return df.filter(bycol + keep_list + [new_col_name])
    return df

def get_count_table(df: pd.DataFrame, bycol: list[str], new_col_name: str = 'count') -> pd.DataFrame:
    """Returns table with columns in <bycol> and their count."""
    bycol= get_label_list(df.columns, bycol)
    df = df.reset_index() # add 'index' column that can be used for count
    df = df.groupby(bycol, observed=False).count().reset_index()
    df.rename({'index': new_col_name}, axis=1, inplace=True)
    return df.filter(bycol + [new_col_name])

# DEPRECATED
def DEADadd_col(df: pd.DataFrame, names: list[str]):
    """Add categorical columns after creating bins.
    Possible <names> options: 'date', 'month', 'season', 
    'timeperiod', 'weekday', 'weather', 'cost', 'datetime'"""
    # print("Creating columns:", names)
    # df = df.copy()
    # # check if there's a datetime column or if it must be added
    # datetime_cols = ['datetime', 'date', 'month', 'season', 'timeperiod', 'weekday']
    # check_date = False
    # possible = datetime_cols + ['weather', 'cost']
    # for name in names:
    #     if name in datetime_cols:
    #         check_date = True
    #     if name not in possible:
    #         raise InvalidColError(name)
    # if check_date:
    #     # look for datetime column and mutates df (the copy of the original)
    #     date_cols = get_datetime_col(df)
    #     print("Datetime column found")
    #     if len(date_cols) > 1:
    #         start_label = get_label(df.columns, 'start time')
    #     date_col = date_cols[0]
    # if 'date' in names:
    #     df['date'] = df[date_col].dt.date

    # return df

# MUTATES
def add_col_Periods(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """columns can be 'month', 'weekday', 'season', 'timeperiod.
    Mutates df in-place'"""
    succeeded = []
    possible_date_cols = get_datetime_col(df)
    if len(possible_date_cols) == 0:
        raise NoDateColumnsError
    datetime_col = add_col_Datetime(df)
    if 'month' in columns  and 'month' not in df.columns:
        df['month'] = df[datetime_col].dt.strftime("%B")
        succeeded.append('month')
    if 'weekday' in columns  and 'weekday' not in df.columns:
        df['weekday'] = df[datetime_col].dt.strftime("%A")
        succeeded.append('weekday')
    if 'season' in columns  and 'season' not in df.columns:
        bins = [pd.to_datetime('12/21/2022'), 
                    pd.to_datetime('03/20/2023'), 
                    pd.to_datetime('06/20/2023'), 
                    pd.to_datetime('09/20/2023'),
                    pd.to_datetime('12/20/2023'),
                    pd.to_datetime('12/31/2023')] # right inclusive, include last  
        labels = ["Winter", "Spring", "Summer", "Fall", "Winter"]
        df["season"] = pd.cut(df[datetime_col], bins=bins, include_lowest=True, ordered=False, labels=labels)
        succeeded.append('season')
    if 'timeperiod' in columns: 
        df['time'] = df[datetime_col].dt.time
        bins = [pd.to_datetime('00:00:00').time(), pd.to_datetime('06:00:00').time(), 
                pd.to_datetime('10:00:00').time(), pd.to_datetime('15:00:00').time(),
                pd.to_datetime('19:00:00').time(), pd.to_datetime('23:00:00').time(),
                pd.to_datetime('23:59:59').time()]
        labels = ["Overnight", "AM", "Midday", "PM", "Evening", "Overnight"]
        df["timeperiod"] = pd.cut(df['time'], bins=bins, include_lowest=True, ordered=False, labels=labels)
        df.drop(['time'], axis=1, inplace=True)
        succeeded.append('timeperiod')
    return succeeded

def add_col_Date(df: pd.DataFrame) -> str:
    """Adds columns 'date' to the dataframe"""
    datetime_col = add_col_Datetime(df)
    df['date'] = df[datetime_col].dt.date
    return datetime_col

def add_col_Datetime(df: pd.DataFrame) -> str:
    """Modifies dataframe to make sure there's a valid datetime column
    and returns that column label (or first if there are multiple options)"""
    
    # datetime_col = get_label(df.columns, 'start time') # ensure valid datetime column
    possible_datecols = get_datetime_col(df)
    if len(possible_datecols) == 0:
        raise NoDateColumnsError
    convert_datecol(df, possible_datecols)
    try: 
        datetime_col = get_label(possible_datecols, 'start time')
    except InvalidColError:
        datetime_col = possible_datecols[0]
    return datetime_col

def convert_datecol(df: pd.DataFrame, columns: list[str]):
    """Attempts to convert every column in <columns> to a datetime column
    Assumes all columns in <columns> are in df.columns"""
    for col in columns:
        try:
            if df[col].dtype not in DATETYPES:
                df[col] = pd.to_datetime(df[col])
        except ValueError:
            pass
    return

# HELPER (MUTATES)
def get_datetime_col(df: pd.DataFrame) -> list[str]:
    """Returns list of the columns in <df> that could be datetime objects."""
    date_cols = []
    for col in df.columns:
        correct_name = 'date' in col.lower() or 'time' in col.lower()
        correct_type = df[col].dtype in DATETYPES or df[col].dtype == 'object'
        if correct_name and correct_type:
            date_cols.append(col)
    return date_cols

# MUTATE
def add_col_Weather(df: pd.DataFrame) -> str: 
    bins = [-16, 0, 5, 15, 30]
    labels = ["Freezing", "Cold", "Cool", "Warm"]
    try: 
        col_name = get_label(df.columns, 'mean temp')
        df["temp_range"] = pd.cut(df[col_name], bins=bins, 
                                  include_lowest=True, labels=labels)
    except InvalidColError:
        print("This dataframe does not have weather-type columns")
    return 'temp_range'

# MUTATE
def add_col_Cost(df: pd.DataFrame) -> str:
    """Adds cost column to the dataframe"""
    user_col, dur_col = get_label(df.columns, 'user type'), get_label(df.columns, 'trip duration')
    df['cost'] = df.apply(lambda x : calculate_cost(x, user_col, dur_col), axis=1)
    return 'cost'

# pure helper 
def calculate_cost(row, user_col: str, dur_col: str):
    user_type = row[user_col]
    if user_type == 'Casual Member':
        return 1 + 0.12*row[dur_col]/60 
    return 0.3 # user is Annual Member

# pure
def station_merge_on_trip(trips: pd.DataFrame, stations: pd.DataFrame, remove_extra_col: bool = True) -> pd.DataFrame:
    """Merges two dataframes based on the station id column.
    <tr> represents trip data, <st> represents station data
    If <od> passed, returns dataframe with each station's n. of origins and destinations.
    Otherwise, doesn't modify the <tr> dataframe and only appends columns  
    with station information for the origin and destination stations"""
    # rename creates a copy
    if remove_extra_col:
        to_drop = ['physical_configuration', 'altitude', 'address', 'is_charging_station', 'rental_methods', 'groups', 'post_code']
        stations.drop(to_drop, axis=1, inplace=True)
    
    st_orig = stations.rename(renamer(stations.columns, '_orig', 'station_id'), axis=1) # don't rename station_id yet
    st_dest = stations.rename(renamer(stations.columns, '_dest', 'station_id'), axis=1)
    
    orig_id_label = get_label(trips.columns, 'start station id')
    dest_id_label = get_label(trips.columns, 'end station id')
    st_orig[orig_id_label] = st_orig['station_id']
    st_dest[dest_id_label] = st_dest['station_id']
    st_dest.drop('station_id', axis=1, inplace=True)
    st_orig.drop('station_id', axis=1, inplace=True)
    
    output = pd.merge(st_orig, trips, on=orig_id_label, how='left')
    output = pd.merge(st_dest, output, on=dest_id_label, how='left')
    output.dropna(subset=['lat_orig'], axis=0, how='any', inplace=True)

    # dead code
    # output = pd.merge(tr, st_orig, on=orig_id_label, how='left')
    # output = pd.merge(output, st_dest, on = dest_id_label, how='left')

    return output

def station_merge_on_station_for_od(tr: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """Merges two dataframes based on the station id column.
    <tr> represents trip data, <st> represents station data.
    Left column
    Returns dataframe with each station's n. of origins and destinations.
    Otherwise, doesn't modify the <tr> dataframe and only appends columns  
    with station information for the origin and destination stations"""
    st_orig = st.copy()
    st_dest = st.copy()

    orig_id_label = get_label(tr.columns, 'start station id') # returns label
    dest_id_label = get_label(tr.columns, 'end station id')
    st_orig[orig_id_label] = st_orig['station_id']
    st_dest[dest_id_label] = st_dest['station_id']
    
    orig_count_label = 'count_orig'
    dest_count_label = 'count_dest'
    orig_count = get_col_count(tr, ['start station id'], orig_count_label)
    dest_count = get_col_count(tr, ['end station id'], dest_count_label)
    
    st_orig = pd.merge(st_orig, orig_count, on=orig_id_label, how='left')
    st_dest = pd.merge(st_dest.filter(['station_id', dest_id_label]), dest_count, on=dest_id_label, how='left')

    output = pd.merge(st_orig, st_dest, on='station_id', how='outer')
    output.fillna({orig_count_label: 0, dest_count_label: 0}, inplace=True)
    output[orig_count_label] = output[orig_count_label].astype(int)
    output[dest_count_label] = output[dest_count_label].astype(int)
    output.drop([orig_id_label, dest_id_label], inplace=True, axis=1)
    
    return output

# helper
def renamer(keys: Iterable, addition: str, avoider: str) -> dict:
    """Renames labels in <keys> (except <avoider>) by combining the 
    label with <addition>. Returns a dictionary."""
    namer = {}
    for key in keys:
        if key != avoider: 
            namer[key] = key + addition
    return namer

# pure
def merge_on_date(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Returns a new dataframe that represents <df1> and <df2> merged  
    on their date columns"""
    add_col_Date(df1)
    add_col_Date(df2)
    df = pd.merge(df1, df2, on='date')
    return df


# pandas commands
# tester = np.arange(len(df))
# if (df['index'] == tester).all():
#     df.drop('index', axis=1, inplace=True)

# if 'trip_count' in names and 'trip_count' not in df.columns:
#     tripid_label = get_label(df.columns, 'trip id')
#     df['trip_count'] = df.groupby([tripid_label])[tripid_label].transform('count')

# df['count'] = df.groupby('group').cumcount()+1
# df['count'] = df.groupby('group')['group'].transform('count')



# dead code from get_col_count()
        # for col in df.columns:
        #     if df[col].nunique() == len(df) or col not in bycol:
        #         rename = col # find column to mutate
        # if not rename:
        #     raise NoIndexTypeColumn()

# dead merge_on
# def merge_on(df1: pd.DataFrame, df2: pd.DataFrame, oncol: str, how: str = 'left') -> pd.DataFrame:
#     """Returns a new dataframe that represents <df1> and <df2> merged on 
#     the column <oncol> using the method <how>."""
    
#     if oncol not in df1.columns:
#         df1 = add_col(df1, [oncol])
#     if oncol not in df2.columns:
#         df2 = add_col(df2, [oncol])
#     if how == 'left_on':
#         df = pd.merge(df1, df2, left_on=oncol, right_on=oncol)
#     else:
#         df = pd.merge(df1, df2, on=oncol, how=how)
#     # print("DF1 info")
#     # df1.info()
#     # print("DF2 info")
#     # df2.info()
#     # print("MERGED info")
#     # df.info()
#     return df