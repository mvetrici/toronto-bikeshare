import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pd_helpers import get_label, add_col_Datetime, add_col_Periods, \
    add_col_Date, get_label_list, InvalidColError, get_count_table

TRIPS = "bikeshare-ridership-2023"
JANTRIPS = r"Bike share ridership 2023-01.csv"
DATA = 'other-datasets-2023'
COLOURS = ['lightsteelblue', 'bisque', 'turquoise', 'khaki', 'salmon', 'peachpuff', 'pink', 'lightcoral', 'thistle']

# theoretically works
def duration_distribution(df: pd.DataFrame, 
                          most_common_pairs: pd.DataFrame, 
                          title: str = '', 
                          group_by_period: bool = False):
    """Plots distribution of trip durations among n 
    most_common_pairs of stations. Does not modify 
    dataframes. df is trip-station data"""    
    df = df.copy()
    if group_by_period:
        add_col_Periods(df, ['timeperiod']) # copy was made so original is not modified
        df = df.loc[(df['timeperiod'] == 'AM') | (df['timeperiod'] == 'PM')]
    durations = df.filter(get_label_list(df.columns, ['start station id', 'end station id', 'trip duration min'])) # 'timeperiod']))
    # Find n most common station pairs
    # duration_counts = get_count_table(durations, new_col_name='count', bycol=['start station id', 'end station id']) #, keep=['trip duration min'])
    # check = duration_counts.sort_values('count', ascending=False)

    o = list(most_common_pairs['Start_Station_Id'])#.iloc[0:n])
    d = list(most_common_pairs['End_Station_Id'])#.iloc[0:n])
    num_plots = len(o)

    # Make historgram
    fig, axs = plt.subplots(num_plots//2, 2, figsize=(6, 10)) 
    for i in range(num_plots):
        start, end = o[i], d[i]
        hist_i = durations.loc[lambda durations: (durations['Start_Station_Id'] == start) & (durations['End_Station_Id'] == end)]
        # if groupby:
        #     data = hist_i[['Trip_Duration_min', groupby]]
        #     data.to_csv('test_w_gpt.csv', index=False)
        # else:
        data = hist_i['Trip_Duration_min']
        
        axs[i//2][i%2].hist(data, bins=30, alpha=0.5, color='b', label=f"{start}-{end}")
        axs[i//2][i%2].set_title(f'{start} and {end}')
        # axs[i%2][int(i in [0, 1])].set_xlabel('Duration (min)')
        # axs[i//2][i%2].set_ylabel('Frequency')
        # axs[i].legend()
    
    # plt.legend()
    if title:
        title = ' for month: ' + title
    plt.suptitle("Durations (mins)" + title) # plt.tight_layout()
    plt.ylabel("Frequency")
    plt.show()

# in progress
def dura_dist_oneplot(df: pd.DataFrame, num_pairs: int = 4, groupby: str = ''):
    """Plots ONE layered distribution of trip durations between <num_pairs> most common station pairs.
    Does not modify dataframes.
    If <num_pairs> > 1, will create new windows.
    df must represent trips, and include columns: start id, end id, trip duration"""    
    df = df.copy()
    if groupby and groupby not in df.columns:
        add_col_Periods(df, [groupby])
    df_cols = get_label_list(df.columns, ['start station id', 'end station id', 'trip duration min', groupby])
    durations = df.filter(df_cols)
    start_col, end_col, trip_dur = df_cols[0], df_cols[1], df_cols[2]
    
    # Find 4 most common station pairs
    duration_counts = get_count_table(durations, new_col_name='count', bycol=['start station id', 'end station id']) #, keep=['trip duration min'])
    check = duration_counts.sort_values('count', ascending=False)
    o = list(check[start_col].iloc[0:num_pairs]) 
    d = list(check[end_col].iloc[0:num_pairs])

    # Make histogram 
    for i in range(len(o)):
        plt.figure(figsize=(10, 6))
        start, end = o[i], d[i]
        # hist_i = durations.loc[lambda durations: (durations[start_col] == start) & (durations[end_col] == end)]
        hist_i = durations.loc[lambda durations: (durations[start_col] == 7076) & (durations[end_col] == 7203)]
        if groupby:
            j = 0
            min_, max_ = min(hist_i[trip_dur]), max(hist_i[trip_dur])
            bins = np.arange(min_, max_ + 1, 0.5)
            for label in hist_i[groupby].unique():
                data_i = hist_i.loc[hist_i[groupby] == label][trip_dur]
                mean_ = round(data_i.mean(), 2)
                plt.hist(data_i, bins=bins, density=True, alpha=0.5, label=f"{label}: {mean_} min", color=COLOURS[j]) # type: ignore
                j += 1
        else:
            data = hist_i[trip_dur]
            plt.hist(data, bins=30, alpha=0.5, label=f"{start}-{end}")
    
        plt.ylabel('Frequency')
        plt.title(f'Distribution of trip durations (min) {start}-{end}')
        plt.legend()
    plt.show()


def minutes_since_midnight(timestamp, interval_size: int = 60) -> float:
    """Returns number of minutes between timestamp and midnight
    OLD: If interval_size is specified (in minutes), returns the number
    of intervals between timestamp and midnight."""
    
    midnight = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    min_since_midnight = (timestamp - midnight).total_seconds() / 60
    # num_interval_per_day = 24 * (60//interval_size)
    # for 60-min intervals: (min_since_midnight//60)*60 
    return (min_since_midnight // interval_size) * interval_size

def trips_per_interval(df: pd.DataFrame, datetime_col: str, interval_size: int = 60) -> pd.DataFrame: 
    """add "<interval_size>_minute" column which represents the number of
    <interval_size> length intervals from 12 am (regardless of day)
    Note: interval_size default = 60 min (1 hour)
    *df* must have a "start time" column"""
    interval_label = f"{interval_size}_minute"
    max_value = 24*60

    # filter for weekdays, then add new colum
    df[interval_label] = df[datetime_col].apply(minutes_since_midnight, args=(interval_size,)).astype(int)
    grouped_df = get_count_table(df, bycol=[interval_label], new_col_name='count')
    grouped_df['count'] = grouped_df['count'] / interval_size
    merger = pd.DataFrame({interval_label: np.arange(0, max_value, interval_size)})
    ret = grouped_df.merge(right=merger, how='right', on=interval_label).fillna(0)

    return ret

def trips_per_interval_stacker(df: pd.DataFrame, interval_sizes: list[int], weekdays: str = '', extra_label: str = ''):
    """Plots multiple lines from trips_per_interval(), by user type and interval size. 
    *df* must have a "start time" column. 
    *weekdays* can be 'weekday' or 'weekend'
    """
    # split data into two, one for each groupby group
    max_value = 24*60
    
    start_time = add_col_Datetime(df) # finds Datetime column
    if start_time != get_label(df.columns, 'start time'):
        raise InvalidColError('start time')
    
    if weekdays == 'weekday':
    # filter for weekdays only
        df = df.loc[df[start_time].dt.weekday < 5]
    elif weekdays == 'weekend':
        df = df.loc[df[start_time].dt.weekday > 4]
    else:
        print("Showing all trips, not ignoring weekday or weekend")

    # separate by User Type
    casual_data = df.loc[df['User_Type'] == 'Casual Member']
    annual_data = df.loc[df['User_Type'] == 'Annual Member']
    interval_sizes.sort()
    fig, ax = plt.subplots(figsize=(11, 14))
    for int_size in interval_sizes:
        # interval_label = f"{int_size}_minute"
        casual_i = trips_per_interval(casual_data, start_time, int_size)
        x = casual_i[casual_i.columns[0]]
        y = casual_i[casual_i.columns[1]]
        print(x)
        print(y)
        plt.plot(x, y, '-', label=f"Casual, {int_size}-min intervals", alpha=0.5)
        # legend.append(f"Casual, {int_size}-min intervals")
        if len(annual_data) > 1:
            annual_i = trips_per_interval(annual_data, start_time, int_size)
            # plt.plot(annual_i[annual_i.columns[0]], annual_i[annual_i.columns[1]], '-', label=f"Annual, {int_size}-min intervals")
        else:
            print("no data for annual members")

    plt.ylabel('Trips per minute')
    labels = []
    nums = np.arange(0, max_value, 60)
    for num in nums:
        if num == 0:
            labels.append('12am') 
        elif num <= 12*60:
            labels.append(str(num//60) + 'am')
        else:
            labels.append(str(num//60 - 12) + 'pm')
    plt.xticks(nums, labels=labels)
    plt.xlabel('Hour')
    plt.legend() # legend
    if extra_label:
        extra_label += ', '
    plt.title(f'Trips taken in 24 hours ({extra_label}{weekdays})')

def run_dura_dist_oneplot(trip_df: pd.DataFrame):
    """trip_df represents trip data"""
    df_casual = trip_df.loc[trip_df['User_Type'] == 'Casual Member']
    df_annual = trip_df.loc[trip_df['User_Type'] == 'Annual Member']
    dura_dist_oneplot(df_casual, num_pairs=1, groupby='timeperiod')
    dura_dist_oneplot(df_annual, num_pairs=1, groupby='timeperiod')

def trips_per_day(trip_df: pd.DataFrame, use_custom_label: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns trips per day. *df* must have a column that can be coerced
    to a datetime dtype."""
    start_time = add_col_Date(trip_df) # finds Datetime column
    casual_data = trip_df.loc[trip_df['User_Type'] == 'Casual Member']
    annual_data = trip_df.loc[trip_df['User_Type'] == 'Annual Member']
    if use_custom_label:
        new_col_name = trip_df[start_time].dt.strftime("%B").unique()[0]
    else:
        new_col_name = ''
    casual_counts = get_count_table(casual_data, ['date'], new_col_name=new_col_name)
    annual_counts = get_count_table(annual_data, ['date'], new_col_name=new_col_name)
    merger = pd.DataFrame({'date': list(trip_df['date'].unique())})
    casual_counts = casual_counts.merge(right=merger, how='right', on='date').fillna(0)
    annual_counts = annual_counts.merge(right=merger, how='right', on='date').fillna(0)
    print(casual_counts)
    print(annual_counts)
    return (casual_counts, annual_counts)

def make_plot(dfs: tuple[pd.DataFrame, pd.DataFrame], legend: list[str]):
    plt.figure(figsize=(8, 10))
    for i in range(len(dfs)):
        x = dfs[i][dfs[i].columns[0]]
        y = dfs[i][dfs[i].columns[1]]
        plt.plot(x, y, '-', label=legend[i], alpha=0.5)
    plt.legend()
    plt.show()

def make_plot_from_df(df: pd.DataFrame, x_axis: str, y_axis: str, type_: str = '-'):
    """x and y must be valid columns in df"""
    plt.figure(figsize=(8, 10))
    x = df[x_axis]
    y = df[y_axis]
    plt.plot(x, y, type_) #label=legend[i], alpha=0.5)
    plt.legend()
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(f"{y_axis} vs {x_axis}")
    plt.show()



if __name__ == '__main__':
    pass