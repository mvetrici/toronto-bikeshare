import pandas as pd
import numpy as np
import os
import re
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

# helper
def get_folder_paths(folder_name: str) -> dict:
    folder_path = os.path.abspath(folder_name)
    folder_dict = {}
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            data_path = os.path.join(folder_path, file)
            folder_dict[file] = data_path
    return folder_dict

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
        precip_label = get_label(df, 'total precip mm')
        df.drop(['Longitude_x', 'Latitude_y'], axis=1, inplace=True)
        df["precip"] = (df[precip_label] > 0)
        df['precip'] = df['precip'].astype(int)
    except InvalidColError:
        pass

    try: # reformat trip data and filter out outlier trips
        duration_label, startid_label, endid_label = get_label_list(df, ['Trip_Duration', 'Start_Station_Id', 'End_Station_Id'])
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
def get_label_list(df: pd.DataFrame, column_names: list[str]) -> list[str]:
    new_labels = column_names.copy()
    for i in range(len(column_names)):
        new_labels[i] = get_label(df, column_names[i])
    return new_labels

# Pure helper
def get_label(df: pd.DataFrame, label: str) -> str:
    """Finds column in a dataset. Does not add columns."""
    # TODO use regex
    bycol = None
    if label in df.columns: # option 1
        return label
    
    for col in df.columns: # check option 2 or 3 to look for column
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
                  keep: list = None) -> pd.DataFrame:
    """Calls groupby on dataframe and returns a new dataframe with two columns:
    first column is the grouper column, second is the count. 
    <bycol> is a list of one or more labels
    """
    bycol= get_label_list(df, bycol)
    rename = False
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
        keep = get_label_list(df, keep)
        keep_list = [item for item in keep if item in df.columns]
        if keep_list != keep:
            print(f"Could not keep all columns in {keep} because they were not valid columns")
        return df.filter(bycol + keep_list + [new_col_name])
    return df

# pure
def add_col(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    # TODO! split into multiple columns
    """Add categorical columns after creating bins.
    Possible <names> options: 'date', 'month', 'season', 
    'timeperiod', 'weekday', 'weather', 'cost', 'datetime'"""
    print("Creating columns:", names)
    df = df.copy()
    # check if there's a datetime column or if it must be added
    datetime_cols = ['datetime', 'date', 'month', 'season', 'timeperiod', 'weekday']
    check_date = False
    possible = datetime_cols + ['weather', 'cost']
    for name in names:
        if name in datetime_cols:
            check_date = True
        if name not in possible:
            raise InvalidColError(name)
    if check_date:
        # look for datetime column and mutates df (the copy of the original)
        date_cols = get_datetime_col(df)
        print("Datetime column found")
        if len(date_cols) > 1:
            start_label = get_label(df, 'start time')
        date_col = date_cols[0]
    if 'date' in names:
        df['date'] = df[date_col].dt.date
    if 'month' in names  and 'month' not in df.columns:
        df['month'] = df[date_col].dt.strftime("%B")
    if 'weekday' in names  and 'weekday' not in df.columns:
        df['weekday'] = df[date_col].dt.strftime("%A")
    if 'season' in names  and 'season' not in df.columns:
        bins = [pd.to_datetime('12/21/2022'), 
                    pd.to_datetime('03/20/2023'), 
                    pd.to_datetime('06/20/2023'), 
                    pd.to_datetime('09/20/2023'),
                    pd.to_datetime('12/20/2023'),
                    pd.to_datetime('12/31/2023')] # right inclusive, include last  
        labels = ["Winter", "Spring", "Summer", "Fall", "Winter"]
        df["season"] = pd.cut(df[date_col].dt.date, bins=bins, include_lowest=True, ordered=False, labels=labels)

    if ('timeperiod' in names) and start_label: # checks if start_label (= trips dataset)
        bins = [pd.to_datetime('00:00:00').time(), pd.to_datetime('06:00:00').time(), 
                pd.to_datetime('10:00:00').time(), pd.to_datetime('15:00:00').time(),
                pd.to_datetime('19:00:00').time(), pd.to_datetime('23:00:00').time(),
                pd.to_datetime('23:59:59').time()]
        labels = ["Overnight", "AM", "Midday", "PM", "Evening", "Overnight"]
        df["timeperiod"] = pd.cut(df[start_label].dt.time, bins=bins, include_lowest=True, ordered=False, labels=labels)
    
    if 'weather' in names:  
        bins = [-16, 0, 5, 15, 30]
        labels = ["Freezing", "Cold", "Cool", "Warm"]
        col_name = None
        for col in df.columns:
            if 'temp' in col.lower() and 'mean' in col.lower():
                col_name = col
        if col_name is None:
            raise InvalidColError(col_name)
        df["temp_range"] = pd.cut(df[col_name], 
                bins=bins, include_lowest=True, labels=labels)
    
    if 'cost' in names:
        user_col, dur_col = get_label(df, 'user type'), get_label(df, 'trip duration')
        df['cost'] = df.apply(lambda x : calculate_cost(x, user_col, dur_col), axis=1)

    return df

# helper 
def calculate_cost(row, user_col: str, dur_col: str):
    user_type = row[user_col]
    if user_type == 'Casual Member':
        return 1 + 0.12*row[dur_col]/60 
    return 0.3 # user is Annual Member

# pure
def station_merge(tr: pd.DataFrame, st: pd.DataFrame, od: bool = False) -> pd.DataFrame:
    """Merges two dataframes based on the station id column.
    <tr> represents trip data, <st> represents station data
    If <od> passed, returns dataframe with each station's n. of origins and destinations.
    Otherwise, doesn't modify the <tr> dataframe and only appends columns  
    with station information for the origin and destination stations"""
    if not od:
        st_orig = st.rename(renamer(st.columns, '_orig', 'station_id'), axis=1) # don't rename station_id yet
        st_dest = st.rename(renamer(st.columns, '_dest', 'station_id'), axis=1)
    else: # if od is passed
        st_orig = st.copy()
        st_dest = st.copy()
    orig_id_label = get_label(tr, 'start station id') # returns label
    dest_id_label = get_label(tr, 'end station id')
    st_orig[orig_id_label] = st_orig['station_id']
    st_dest[dest_id_label] = st_dest['station_id']
    
    if od:
        orig_count_label = 'count_orig'
        dest_count_label = 'count_dest'
        orig_count = get_col_count(tr, ['start station id'], orig_count_label)
        dest_count = get_col_count(tr, ['end station id'], dest_count_label)
        # print('check int or float')
        # print(dest_count)
        
        st_orig = pd.merge(st_orig, orig_count, on=orig_id_label, how='left')
        st_dest = pd.merge(st_dest.filter(['station_id', dest_id_label]), dest_count, on=dest_id_label, how='left')

        output = pd.merge(st_orig, st_dest, on='station_id', how='outer')
        output.fillna({orig_count_label: 0, dest_count_label: 0}, inplace=True)
        output[orig_count_label] = output[orig_count_label].astype(int)
        output[dest_count_label] = output[dest_count_label].astype(int)
        output.drop([orig_id_label, dest_id_label], inplace=True, axis=1)
        return output
    
    output = pd.merge(tr, st_orig, on=orig_id_label, how='left')
    output = pd.merge(output, st_dest, on = dest_id_label, how='left')

    # output = pd.merge(tr, st_orig, on=name, how='left')
    # output.drop('station_id', axis=1, inplace=True)
    # st_dest.drop('station_id', axis=1, inplace=True)

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
def merge_on(df1: pd.DataFrame, df2: pd.DataFrame, oncol: str, how: str = 'left') -> pd.DataFrame:
    """Returns a new dataframe that represents <df1> and <df2> merged on 
    the column <oncol> using the method <how>."""
    
    if oncol not in df1.columns:
        df1 = add_col(df1, [oncol])
    if oncol not in df2.columns:
        df2 = add_col(df2, [oncol])
    if how == 'left_on':
        df = pd.merge(df1, df2, left_on=oncol, right_on=oncol)
    else:
        df = pd.merge(df1, df2, on=oncol, how=how)
    # print("DF1 info")
    # df1.info()
    # print("DF2 info")
    # df2.info()
    # print("MERGED info")
    # df.info()
    return df

# HELPER (MUTATES)
def get_datetime_col(df: pd.DataFrame) -> list[str]:
    """Returns list of the column labels that are datetime objects.
    Converts columns to datetime objects if they're not.
    (eventually consider standardizing column names)"""
    datetypes = ['datetime64[ns]', '<M8[ns]']
    date_cols = []
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            if df[col].dtype in datetypes:
                date_cols.append(col)
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_cols.append(col)
                except ValueError:
                    pass
    if len(date_cols) == 0:
        raise NoDateColumnsError()
    return date_cols


# pandas commands
# tester = np.arange(len(df))
# if (df['index'] == tester).all():
#     df.drop('index', axis=1, inplace=True)

# if 'trip_count' in names and 'trip_count' not in df.columns:
#     tripid_label = get_label(df, 'trip id')
#     df['trip_count'] = df.groupby([tripid_label])[tripid_label].transform('count')

# df['count'] = df.groupby('group').cumcount()+1
# df['count'] = df.groupby('group')['group'].transform('count')



# dead code from get_col_count()
        # for col in df.columns:
        #     if df[col].nunique() == len(df) or col not in bycol:
        #         rename = col # find column to mutate
        # if not rename:
        #     raise NoIndexTypeColumn()