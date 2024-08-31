import pandas as pd
import os, re
from typing import Iterable

MAX_LENGTH = 120 # time in minutes of longest possible trip
DATETYPES = ['datetime64[ns]', '<M8[ns]']

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


def df_from_file(path: str, encoding: str = 'cp1252', drop: list[str] = []) -> pd.DataFrame:
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

    try:
        # remove BOM and columns with mostly NAs
        col_drop = ['rental_uris', "obcn", "short_name", 'nearby_distance']
        col_drop += ['_ride_code_support', 'Climate_ID', 'Station_Name'] 
        col_drop += drop
        col_drop += ['altitude', 'address',	'capacity', 'is_charging_station', 'rental_methods', 'groups', 'post_code', 'is_valet_station', 'cross_street']

        for col in df.columns:
            col1 = col.encode('cp1252').decode('utf-8-sig', 'ignore')
            col1 = re.sub(r'\W+', '', col1)
            df.rename(columns={col: col1}, inplace=True)
            
            # remove unhelpful columns or columns with mostly NAs 
            remove = col1 in col_drop or 'flag' in col1.lower() or float(df[col1].isna().sum()) > 0.5*len(df) or 'days' in col1.lower() or 'gust' in col1.lower()
            if remove:
                df.drop(col1, axis=1, inplace=True)

            # remove rows with NAs in certain columns
            row_drop = ["End_Station_Id", 'Min_Temp_C', 'Mean_Temp_C', 'Max_Temp_C']
            dropna = col1 in row_drop
            if dropna: 
                df.dropna(subset=[col1], axis=0, how='any', inplace=True)
    except TypeError:
        pass
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
        filtering, shortest = " (with filtering)", len(df)
    except InvalidColError:
        pass
    
    print(orig_length - shortest, f"observations were removed{filtering}. Final length: {len(df)} rows.")
    df.reset_index(inplace=True, drop=True)
    
    return df

def get_count_table(df: pd.DataFrame, bycol: list[str], new_col_name: str = 'count', keep: list[str] = []) -> pd.DataFrame:
    """Returns table with rows in *df* grouped by *bycol* and their count."""
    columns = df.columns
    bycol= get_label_list(columns, bycol)
    df = df.reset_index() # add 'index' column that can be used for count
    df_counts = df.groupby(bycol, observed=False).count().reset_index()
    df_counts.rename({'index': new_col_name}, axis=1, inplace=True)
    df_counts = df_counts[bycol + [new_col_name]] 
    if keep:
        keep = get_label_list(columns, keep)
        df_counts = pd.merge(left=df_counts, right=df[bycol + keep], on=bycol[0], how='left')
        df_counts.drop_duplicates(inplace=True)
    return df_counts

# Pure
def get_label_list(possible_labels: Iterable, testing_labels: list[str]) -> list[str]:
    new_labels = testing_labels.copy()
    for i in range(len(testing_labels)):
        new_labels[i] = get_label(possible_labels, testing_labels[i])
    return new_labels

# Pure helper
def get_label(possible_labels: Iterable, label: str) -> str:
    """Finds corresponding str <label> among the options
     in <possible_labels>. Does not modify possible_labels."""

    # option 1: label is exactly already in dataframe
    if label in possible_labels:
        return label
    
    # option 2: search through options
    splitter = ' ' if ' ' in label else '_'
    label_list = label.split(splitter)
    pattern = r'.*' 
    for label_word in label_list:
        pattern += label_word + r'.*' 
    
    for possible_label in possible_labels:
        current_match = re.search(pattern, possible_label, flags=re.I)
        if current_match:
            return possible_label
    raise InvalidColError(label)


# FUNCTIONS TO ADD COLUMNS

# MUTATES
def add_col_Periods(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """columns can be 'month', 'weekday', 'season', 'timeperiod.'
    Mutates df in-place"""
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
        df["season"] = pd.cut(df[datetime_col], bins=bins, include_lowest=True, ordered=False, labels=labels) # type:ignore
        succeeded.append('season')
    if 'timeperiod' in columns: 
        df['time'] = df[datetime_col].dt.time
        bins = [pd.to_datetime('00:00:00').time(), pd.to_datetime('06:00:00').time(), 
                pd.to_datetime('10:00:00').time(), pd.to_datetime('15:00:00').time(),
                pd.to_datetime('19:00:00').time(), pd.to_datetime('23:00:00').time(),
                pd.to_datetime('23:59:59').time()]
        labels = ["Overnight", "AM", "Midday", "PM", "Evening", "Overnight"]
        df["timeperiod"] = pd.cut(df['time'], bins=bins, include_lowest=True, ordered=False, labels=labels) # type:ignore
        df.drop(['time'], axis=1, inplace=True)
        succeeded.append('timeperiod')
    return succeeded

def add_col_Date(df: pd.DataFrame) -> str:
    """Adds columns 'date' to the dataframe (mutates)"""
    datetime_col = add_col_Datetime(df)
    df['date'] = df[datetime_col].dt.date
    return datetime_col

def add_col_Datetime(df: pd.DataFrame) -> str:
    """Modifies dataframe to make sure there's a valid datetime column
    and returns that column label (or first if there are multiple options)"""
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


# MERGING FUNCTIONS

# pure
def station_merge_on_trip(trips: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Merges two dataframes (trips and stations) based on the station id column.
    (appends columns with station information for origin and destination stations."""
    
    orig_id_label = get_label(trips.columns, 'start station id')
    dest_id_label = get_label(trips.columns, 'end station id')

    st_orig = stations.rename(
        renamer(stations.columns, '_orig', 'station_id', orig_id_label), 
        axis=1) # rename columns so merging is easier 
    st_dest = stations.rename(
        renamer(stations.columns, '_dest', 'station_id', dest_id_label), 
        axis=1)
    
    output = pd.merge(st_orig, trips, on=orig_id_label, how='left')
    output = pd.merge(st_dest, output, on=dest_id_label, how='left')
    output.dropna(subset=['lat_orig'], axis=0, how='any', inplace=True)

    return output

def station_merge_on_station_for_od(tr: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """Merges two dataframes based on the station id column.
    *tr* represents trip data, *st* represents station data.
    Returns dataframe with each station's n. of origins and destinations.
    Doesn't modify the <tr> dataframe and only appends columns  
    with station information for the origin and destination stations"""
    st_orig = st.copy()
    st_dest = st.copy()

    orig_id_label = get_label(tr.columns, 'start station id') # returns label
    dest_id_label = get_label(tr.columns, 'end station id')
    st_orig[orig_id_label] = st_orig['station_id']
    st_dest[dest_id_label] = st_dest['station_id']
    
    orig_count_label = 'count_orig'
    dest_count_label = 'count_dest'

    orig_count = get_count_table(tr, ['start station id'], orig_count_label)
    dest_count = get_count_table(tr, ['end station id'], dest_count_label)
    
    st_orig = pd.merge(st_orig, orig_count, on=orig_id_label, how='left')
    st_dest = pd.merge(st_dest.filter(['station_id', dest_id_label]), dest_count, on=dest_id_label, how='left')

    output = pd.merge(st_orig, st_dest, on='station_id', how='outer')
    output.fillna({orig_count_label: 0, dest_count_label: 0}, inplace=True)
    output[orig_count_label] = output[orig_count_label].astype(int)
    output[dest_count_label] = output[dest_count_label].astype(int)
    output.drop([orig_id_label, dest_id_label], inplace=True, axis=1)
    
    return output

def station_merge_on_id(trip: pd.DataFrame, station: pd.DataFrame):
    """*trip* must have ONE column with the name station id"""
    for col in trip.columns:
        if bool(re.search('station', col, flags=re.I)) and bool(re.search('id', col, flags=re.I)):
            station_col = col
    for col in station.columns:
        if bool(re.search('station', col, flags=re.I)) and bool(re.search('id', col, flags=re.I)):
            print("found column to rename")
            station.rename({col: station_col}, axis=1, inplace=True)
        else:
            InvalidColError("station id")
    print(station)
    # merged = pd.merge(trip, station, on=station_col, how='left')
    return pd.merge(station, trip, on=station_col, how='left')

# helper
def renamer(keys: Iterable, addition: str, replaced: str, replacer: str) -> dict:
    """Renames labels in *keys* by appending the label 
    with *addition*. Returns a dictionary. *replaced* must be valid
    in *keys*"""
    if replaced not in keys:
        raise InvalidColError(replaced)
    namer = {}
    for key in keys:
        namer[key] = replacer if key == replaced else key + addition
    return namer

# pure
def merge_on_date(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Returns a new dataframe that represents <df1> and <df2> merged  
    on their date columns"""
    add_col_Date(df1)
    add_col_Date(df2)
    df = pd.merge(df1, df2, on='date')
    return df