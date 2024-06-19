import pandas as pd
import numpy as np
import os
import re

class IncompatibleDataframes(Exception):
    pass

class NoDateColumnsError(Exception):
    pass

class InvalidColError(Exception):
    pass

class NotSureWhatToCountError(Exception):
    pass

MAX_LENGTH = 120

# pandas commands
# tester = np.arange(len(df))
# if (df['index'] == tester).all():
#     df.drop('index', axis=1, inplace=True)

# if 'trip_count' in names and 'trip_count' not in df.columns:
#     tripid_label = get_label(df, 'trip id')
#     df['trip_count'] = df.groupby([tripid_label])[tripid_label].transform('count')

# global helper
def get_folder_paths(folder_name: str) -> dict:
    folder_path = os.path.abspath(folder_name)
    folder_dict = {}
    for file in os.listdir(folder_path):
        data_path = os.path.join(folder_path, file)
        folder_dict[file] = data_path
    return folder_dict

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
    col_drop = ['rental_uris', "obcn", "short_name", 'nearby_distance', '_ride_code_support', 'Climate_ID', 'Station_Name']
    for col in df.columns:
        col1 = col.encode('cp1252').decode('utf-8-sig', 'ignore')
        col1 = re.sub(r'\W+', '', col1)
        df.rename(columns={col: col1}, inplace=True)
        
        # remove unhelpful columns or columns with mostly NAs 
        remove = col1 in col_drop or 'flag' in col1.lower() or int(df[col1].isna().sum()) > 0.5*len(df) or 'days' in col.lower() or 'gust' in col.lower()
        if remove:
            df.drop(col1, axis=1, inplace=True)

        # remove rows with NAs in certain columns
        row_drop = ["End_Station_Id", 'Min_Temp_C', 'Mean_Temp_C', 'Max_Temp_C']
        dropna = col1 in row_drop
        if dropna: 
            df.dropna(subset=[col1], axis=0, how='any', inplace=True)

    print("Cleaned length:", len(df))

    try:
        precip_label = get_label(df, 'total precip mm')
        df.drop('Longitude_x', axis=1, inplace=True)
        df["precip"] = (df[precip_label] > 0)
        df['precip'] = df['precip'].astype(int)
    except InvalidColError:
        pass

    try:
        duration_label = get_label(df, 'Trip_Duration')
        startid_label = get_label(df, 'Start_Station_Id')
        endid_label = get_label(df, 'End_Station_Id')
        df[endid_label] = df[endid_label].astype(int)
        df["Trip_Duration_(min)"] = round(df[duration_label]/60, 2)
        df = df.loc[(df["Trip_Duration_(min)"] <= MAX_LENGTH) 
                    & (df["Trip_Duration_(min)"] >= 2)]
        df = df.loc[df[startid_label] != df[endid_label]]
        print("Filtered length:", len(df))
    except InvalidColError:
        pass
    
    df.reset_index(inplace=True, drop=True)
    return df


# Pure (returns new dataframe object)
def get_label(df: pd.DataFrame, tester: str) -> str:
    """Finds column in a dataset"""
    bycol = None
    if tester in df.columns: # option 1
        bycol = tester
    
    for col in df.columns: # check option 2 or 3 to look for column
        splitter = ' '
        if '_' in tester:
            splitter = '_'
        test_list = [int(item in col.lower()) for item in tester.split(splitter)]
        
        splitter = ' '
        if '_' in col:
            splitter = '_'
        if np.array(test_list).all() and len(test_list) == len(col.split(splitter)):
            bycol = col
    if bycol:
        return bycol
    else:
        raise InvalidColError("Invalid column name")

def get_col_count(df: pd.DataFrame, bycol: list[str], new_col_name: str = 'count', new: bool = False) -> pd.DataFrame:
    """Calls groupby on dataframe and returns a new dataframe with two columns
    first column is the grouper column, second is the count. 
    <bycol> is a column or a list of labels

    """

    for i in range(len(bycol)):
        bycol[i] = get_label(df, bycol[i])
    for col in df.columns:
        if df[col].nunique() == len(df):
            keep = col
    if not keep:
        raise NotSureWhatToCountError("No column found")
    
    if new:
        df = df.groupby(bycol, observed=False).count().reset_index()
        # sort=False, dropna=False
        df.rename({keep: new_col_name}, axis=1, inplace=True)
        return df.filter(bycol + [new_col_name])
        
    else: 
        df[new_col_name] = df.groupby(bycol, dropna=False, observed=True)[keep].transform('count')
        return df

def add_col(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Add columns after creating bins.
    Names should follow format of xxx_xxx
    Possible options: 'date', 'month', 'season', 'timeperiod', 'weekday', 'weather'"""
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
            print(f"Error in add_col: Column type <{name}> not valid")
    if check_date:
        # look for datetime column and mutates df (the copy of the original)
        date_cols = get_datetime_col(df)
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
            raise InvalidColError(f"<{col_name}> is not a column label.")
        df["temp_range"] = pd.cut(df[col_name], 
                bins=bins, include_lowest=True, labels=labels)
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
def get_datetime_col(df: pd.DataFrame) -> list[str]:
    """Returns name of the columns that are datetime objects.
    Converts columns to datetime objects if they're not.
    (eventually consider standardizing column names?)"""
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
        raise NoDateColumnsError("No date columns were found")
    return date_cols