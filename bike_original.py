import requests
import json
import datetime
import pytz
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import io



WEATHER_PATH = r"C:\Users\mvetr\OneDrive - University of Toronto\Desktop\Summer research\to-city-cen-weather-2023\weather2023.csv"

def get_system_info(system_info_url: str) -> None:
    system_info_url = "https://tor.publicbikesystem.net/ube/gbfs/v1/en/system_information"
    system_info = requests.get(system_info_url).text
    system_info = json.loads(system_info)
    print("Last updated:")
    unix_to_utc = datetime.datetime.utcfromtimestamp(system_info['last_updated'])
    utc_timezone = pytz.timezone('UTC')
    localized = utc_timezone.localize(unix_to_utc)
    toronto_timezone = pytz.timezone('America/Toronto')
    toronto_datetime = localized.astimezone(toronto_timezone)
    print(toronto_datetime)

def read_manipulate(data_path: str) -> pd.DataFrame:
    """Read in CSV file at path <data_path>."""
    return pd.read_csv(data_path, encoding='cp1252')

def clean_data(data_path: str, na_list: list[str] = []) -> pd.DataFrame: 
    """Clean dataframe <df> (and remove NAs in columns in <na_list>).
    Also, replace times with datetime objects and remove BOM.
    Mutating function that returns original length (int).
    """
    df = read_manipulate(data_path)
    # remove na
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace('  ', ' ')
    df.columns = df.columns.str.replace('"', '')
  
    for col in df.columns:
        col1 = col.encode('cp1252').decode('utf-8-sig', 'ignore')
        df.rename(columns={col: col1}, inplace=True)
        if int(get_nas(df[col1])) > 0.5*len(df):
            df.drop(col1, axis=1, inplace=True)

    if "bikeshare" in data_path:
        dropna_list = ["End Station Id"]
        df.dropna(subset=dropna_list, axis=0, how='any', inplace=True)
    if "stations" in data_path: 
        dropcol_list = ['rental_uris', "obcn", "short_name", 'nearby_distance', '_ride_code_support']
        df.drop(labels=dropcol_list, axis=1, inplace=True)
    return df

# NOTE: turn into a mutating function?
def filter_trips(df: pd.DataFrame, max_length: int = 120) -> pd.DataFrame:
    """Returns total number of non-NA trips where <my_data> is a path.
    Add new "Trip Duration (min)" column.
    Remove trips greater than <max_length> min and shorter than 2 min.
    """
    # reformat data values
    df = df.astype({"Trip Id": int, "Start Station Id": int, "End Station Id": int})
    df['Start Time'] = pd.to_datetime(df['Start Time'], format='%m/%d/%Y %H:%M')
    df['End Time'] = pd.to_datetime(df['End Time'], format='%m/%d/%Y %H:%M')
    
    # filter durations
    df["Trip Duration (min)"] = round(df["Trip Duration"]/60, 2)
    df = df.loc[(df["Trip Duration (min)"] <= max_length) 
                      & (df["Trip Duration (min)"] >= 2)]
    
    # remove trips with same starting and ending location
    df = df.loc[df['Start Station Id'] != df['End Station Id']]
    
    return df
    
def weather_val(row):
    temp, v = row['Mean Temp (°C)'], row['Spd of Max Gust (km/h)']
    if temp < 10 and v > 4.8:
        return 13.12 + temp*0.6125 - 11.37*v**0.16 + 0.3965*temp*v**0.16
    return row['Max Temp (°C)'] 

def filter_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Filter NA values and remove extra columns"""
    df.dropna(subset=['Min Temp (°C)', 'Mean Temp (°C)', 'Max Temp (°C)'], axis=0, how='any', inplace=True)
    # df['Perceived Temp'] = df.apply(lambda x : weather_val(x), axis=1)
    bins = [-16, 0, 5, 15, 30]
    labels = ["Freezing", "Cold", "Cool", "Warm"]
    df["Temperature Range"] = pd.cut(df["Mean Temp (°C)"], 
                                     bins=bins, include_lowest=True, labels=labels)
    df["Precip"] = (df["Total Precip (mm)"] > 0).astype(int)
    return df

def get_nas(df: pd.DataFrame) -> pd.DataFrame:
    return df.isna().sum()

def get_stats(df: pd.DataFrame, func: str = None, 
              grouping: str = "User Type", 
              col_interest: list[str] = ["Trip Duration (min)"]) -> pd.DataFrame:
    if grouping:
        df = df.groupby(grouping)
    if col_interest:
        df = df[col_interest]
    if func == 'mean':
        return df.mean()
    return df.describe()

def folder_processor(folder_name: str, file_func = None, test: str = 'not test', merge: str ='no') -> pd.DataFrame:
    """Read all the files in the folder <folder_name>.
    and run function <file_func> on them.
    Return a list of the output."""
    output = pd.DataFrame()
    folder_path = os.path.abspath(folder_name)
    folder_paths = [file for file in os.listdir(folder_path)]
    if test == 'test':
        folder_paths = [file for file in os.listdir(folder_path) if file.endswith('08.csv') or file.endswith('09.csv')]
    for file in folder_paths:
        print(f"File '{file}' is being processed.")
        # output.append(count_non_na(os.path.join(folder_path, file)))
        data_path = os.path.join(folder_path, file)
        df = clean_data(data_path)
        if folder_name == "bikeshare-ridership-2023":
            df = filter_trips(df)
            print(f"{file}", "length:", len(df))
        ignore_index, axis = False, 0
        if merge:
            df2 = filter_weather(clean_data(WEATHER_PATH))
            df = merge_df(df, df2, 'date')
        if file_func:  # if a function isn't passed, it will concatenate all files
            df = file_func(df)
            ignore_index, axis = True, 1 
            if output.empty:  # first iteration
                output = df.copy()
            else:
                output = pd.concat([output, df], axis=axis, ignore_index=ignore_index)
        else:
            df = counts_by_col(df)
            if output.empty:  # first iteration
                output = df.copy()
            else:
                # output = pd.merge(output, df, left_index=True, right_index=True)
                output = output.join(df)
                print("MERGED but NOT aggregated")
                print(output)
                output = aggregator(output)
    return output

def aggregator(df: pd.DataFrame) -> pd.DataFrame:
    """Returns aggregated dataframe so only one column"""
    to_drop = list(df.columns) 
    df['Count'] = df.sum(axis=0) 
    df.drop(to_drop, axis=1, inplace=True) 
    return df 

def folder_concat(folder_name: str, test: str = 'not test', merge: str = 'not test') -> pd.DataFrame:
    """Read all the files in the folder <folder_name> and concatenate them.
    """
    output = pd.DataFrame()
    folder_path = os.path.abspath(folder_name)
    folder_paths = [file for file in os.listdir(folder_path)]
    if test == 'test':
        folder_paths = [file for file in os.listdir(folder_path) if file.endswith('08.csv') or file.endswith('09.csv')]
    for file in folder_paths:
        print(f"File '{file}' is being processed.")
        # output.append(count_non_na(os.path.join(folder_path, file)))
        data_path = os.path.join(folder_path, file)
        df = clean_data(data_path)
        if folder_name == "bikeshare-ridership-2023":
            df = filter_trips(df)
            print(f"{file}", "length:", len(df))
        ignore_index, axis = False, 0
        if merge:
            df2 = filter_weather(clean_data(WEATHER_PATH))
            df = merge_df(df, df2, 'date')
        if output.empty:  # first iteration
            output = df.copy()
        else:
            output = pd.concat([output, df], axis=axis, ignore_index=ignore_index) # add ignore_index if concatenating trips data

        # new_name = 
        # output.rename({list(output.columns)[-1]: new_name}, axis=1, inplace=True)
    return output

def trips_processor(data_path: str) -> pd.DataFrame:
    return filter_trips(clean_data(data_path))

def get_unique(df: pd.DataFrame, col: str = "User Type") -> dict:
    return {'keys': len(df[col].unique())}

def counts_by_col(df: pd.DataFrame, col: str = "Season") -> pd.DataFrame:
    print("Creating a new column...")
    # print("JUST COLUMN")
    # print(df['Start Time'])
    # print("FIRST VALID INDEX")
    # print(df.first_valid_index())
    # print("OBJECT I WANT")
    # print(df['Start Time'][df.first_valid_index()])
    month_lab = df['Start Time'][df.first_valid_index()].to_pydatetime().strftime("%B")
    if col == "Month":
        df['Month'] = df["Start Time"].dt.strftime("%B")
    if col == "Weekday":
        df['Weekday'] = df["Start Time"].dt.strftime("%A")
    if col == "Season":
        bins = [pd.to_datetime('12/21/2022'), 
                pd.to_datetime('03/20/2023'), 
                pd.to_datetime('06/20/2023'), 
                pd.to_datetime('09/20/2023'),
                pd.to_datetime('12/20/2023'),
                pd.to_datetime('12/31/2023')] # right inclusive, include last  
        df["Season"] = pd.cut(df["Start Time"].dt.date, bins=bins, include_lowest=True, ordered=False, labels=["Winter", "Spring", "Summer", "Fall", "Winter"])
    if col == "Time Period": 
        bins = [pd.to_datetime('00:00:00').time(), pd.to_datetime('06:00:00').time(), 
                pd.to_datetime('10:00:00').time(), pd.to_datetime('15:00:00').time(),
                pd.to_datetime('19:00:00').time(), pd.to_datetime('23:00:00').time(),
                pd.to_datetime('23:59:59').time()]
        labels = ["Overnight", "AM", "Mid-day", "PM", "Evening", "Overnight"]
        df["Time Period"] = pd.cut(df["Start Time"].dt.time, bins=bins, include_lowest=True, ordered=False, labels=labels)
    if col == 'date':
        if 'date' not in df.columns:
            df['date'] = df['Start Time'].dt.date
    print("Finding labels...")
    # df = df.loc[df[col].notna()] # drop NA values
    sorted = df[col].unique()
    print("column values\n", sorted)
    # implement sorting
    # output = df.groupby(col).count()
    # sorted = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # output['Weekday'] = pd.Categorical(output['Weekday'], sorted)
    print("Grouping and counting...")
    # print("output BEFORE grouping\n", df)
    output = df.groupby(col, sort=False, observed=True).count() # changed from observed=False
    # print("output AFTER grouping\n", output)
    # output = output.reindex(index=sorted, columns=[col, "Trip Id"])[["Trip Id"]]
    output = output.filter(["Trip Id"])
    # print("output AFTER removing columns\n", output)
    # print("type is", type(output))
    output.rename(columns={"Trip Id": month_lab}, inplace=True)
    # print("output AFTER renaming\n", output)
    # output[label] = round(output[label]/sum(output[label]), 4) # to get proportion
    return output

def merge_df(df1: pd.DataFrame, df2: pd.DataFrame, col: str = None) -> pd.DataFrame:
    """df1 should be trips data, df2 should be weather data
    OR df1 should be trips data and df2 should be station data"""
    if col == 'date':
        # df1[col] = pd.to_datetime(df1['Start Time']).dt.date  # convert to datetime object
        if col not in df1.columns:
            df1[col] = df1['Start Time'].dt.date # if already converted to datetime object
        df2[col] = pd.to_datetime(df2['Date/Time']).dt.date
            # print("df1 info:\n", df1.info())
        # print("df2 info:\n", df1.info())
        output = pd.merge(df1, df2, on=col, how='left')
        output.drop(col, axis=1, inplace=True)
        return output 
    if col == 'station id':
        rename_orig = {}
        for item in df2.columns[1:]:
            rename_orig[item] = item + '_orig'
        orig_stations = df2.rename(rename_orig, axis=1)
        print("ORIG STATIONS")
        print(orig_stations)
        orig_stations['Start Station Id'] = orig_stations['station_id']
        # merge on "Start Station Id"
        output = pd.merge(df1, orig_stations, on='Start Station Id', how='left')
        rename_dest = {}
        for item in df2.columns[1:]:
            rename_dest[item] = item + '_dest'
        dest_stations = df2.rename(rename_dest, axis=1)
        dest_stations['End Station Id'] = dest_stations['station_id']
        print("DEST STATIONS")
        print(dest_stations)
        # merge on "End Station Id"
        output = pd.merge(output, dest_stations, on='End Station Id', how='left')
        # output.drop('station_id', axis=1, inplace=True) # both have _x and _y
        return output 
    output = pd.merge(df1, df2, how='outer', left_index=True, right_index=True)
    output.fillna(0, inplace=True)
    # output.reset_index(inplace=True) # converts index into its own column
    return output

def make_bar(df: pd.DataFrame, name: str = "Barplot"):
    # plt.subplots(nrows=3, ncols=4, sharey='row')
    # plt.figure(figsize=(10, 10))
    # x = np.arange(len(df.columns)) 
    # width = 0.5
    # multiplier = 0
    # fig, ax = plt.subplots(layout='constrained')
    # for data_type, value in my_data.items():
    #     offset = width * multiplier
    #     rects = ax.bar(x + offset, height=value, width=width, label=data_type)
    #     ax.bar_label(rects, padding=3)
    #     multiplier += 1
    n_cols = int(len(df.columns))
    figsize = (12, 4)
    # bar_positions = np.arange(len(index_vals)) 
    # Iterate over each column and create a separate bar plot in each subplot
    # index stores each bar label
    if n_cols > 1:
        fig, axes = plt.subplots(nrows=1, ncols=n_cols, figsize=figsize, sharey=True, layout='constrained')
        index_vals = df.index.to_list()
        for i, column in enumerate(df.columns):
            # for j, value in enumerate(df[column]): 
            #         # Shift each bar closer to each other
            #         bar_shift = (j - len(index_vals)/2) * 0.05
            #         axes[i].bar(bar_positions[j] + bar_shift, value, width=0.1, color='skyblue')

            df[column].plot(kind='bar', ax=axes[i], color='skyblue')
            axes[i].set_title(column)
            axes[i].set_xticks(range(len(index_vals)), labels= index_vals) 
            axes[i].set_xticklabels(df.index.to_list())
        fig.suptitle(name)
        # fig.supylabel("Count")
        # fig.supxlabel(df.index.name)
        # ax.set_xticks(x + width, categories)
        # ax.legend(loc='upper left', ncols=3)
        # axes.set_ylim(0, 1.1*max(df.values()))
        plt.tight_layout()
        plt.show()
    
    else:
        df.plot(kind='bar', figsize=figsize, use_index=True, xlabel=df.index.name, rot=0, ylabel=df.columns[0])
        plt.show()

def make_hist(df: pd.DataFrame, value: str = 'Trip Duration (min)'):
    bins=100
    plt.figure(figsize=(5, 2.5))

    # plt.hist([jm_trips.loc[jan_trips["Rained"] == 1]["Mean Temp (C)"], jan_trips.loc[jan_trips["Rained"] == 0]["Mean Temp (C)"]],
    #          bins=bins, alpha=0.5, label=['Rainy days', "Dry days"], stacked = True)
                
    plt.hist(df[value], 
            bins=bins, alpha=0.5, label=value) 

    plt.xlabel('Trip Duration (minutes)')
    plt.ylabel('Frequency')
    plt.title(f'Variation in trip duration')

    mean_dur = df["Trip Duration (min)"].mean()
    med_dur = df["Trip Duration (min)"].median()
    plt.axvline(x = mean_dur, color = 'b', label = f"Mean duration {round(mean_dur, 2)} minutes")
    plt.axvline(x = med_dur, color = 'c', label = f"Median duration {round(med_dur, 2)} minutes")
    plt.legend()
    plt.grid(True)

    plt.show()



# def BROKEN_count_non_na(df: pd.DataFrame) -> dict:
#     """Count NAs and total filtered length in df at <data_path>.
#     Returns a dictionary with the file name, length, and removed NAs"""
#     original = clean_data(df)
#     final = len(df)
#     output = {"final": final, "NAs": original-final}
#     return {}

# dump
  # for col in trips.columns:
    #     i = 0
    #     while not col[i].isalnum():  # need something to deal with i with dots which is technically alnum
    #         i += 1
    #     print(col[i:])
    #     trips.rename(columns={col: col[i:]}, inplace=True)

# bar_labels = ['red', 'blue', '_red', 'orange']
    # ax.bar(x=x, height=list(my_data.values()), label=list(my_data.keys()))
    # optional set colours = bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']
    # ax.set_ylabel('Count')
    # ax.set_title(title)
    # ax.legend(title=title)
    # ax.set_ylim(0, 1.2*max(my_data.values()))

# fall = (pd.to_datetime('09/21/2023'), pd.to_datetime('12/20/2023'))  # month is first
# winter1 = (pd.to_datetime('12/21/2023'), pd.to_datetime('12/31/2023'))
# winter2 = (pd.to_datetime('01/01/2023'), pd.to_datetime('03/20/2023'))
# spring = (pd.to_datetime('03/21/2023'), pd.to_datetime('05/20/2023'))
# summer = (pd.to_datetime('05/21/2023'), pd.to_datetime('09/20/2023'))
# bins = pd.IntervalIndex.from_tuples([(winter2, spring), (spring, summer), (summer, fall), (fall, winter1)])


# folder_processor old code
# if file_func != print:
        #     output_i = file_func(df)
        #     if type(output_i) != dict:  # assume it's a df if not dict
        #         output_i.to_dict()
        #     output = {key: 
        #               output.get(key, 0) + output_i.get(key, 0) for key in set(output_i) | set(output)}
        
        # else:
        #     print(df)
