import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pd_helpers import get_label, add_col, get_col_count, get_label_list
from folderProcessor import folderProcessor

TRIPS = "bikeshare-ridership-2023"
JANTRIPS = r"Bike share ridership 2023-01.csv"
DATA = 'other-datasets-2023'
COLOURS = ['lightsteelblue', 'bisque', 'turquoise', 'khaki', 'salmon', 'peachpuff', 'pink', 'lightcoral', 'thistle']

# in progress
def visualize_cost():
    trips_folder = folderProcessor(TRIPS, 'test')
    other_folder = folderProcessor(DATA)
    df = trips_folder.combine_merge(other_folder).get_df()
    df = add_col(df, ['weather', 'cost'])

    df1 = df.groupby(['Start_Station_Id', 'End_Station_Id'])['cost'].mean().sort_values(ascending=False)
    df2 = df.groupby(['Start_Station_Id', 'End_Station_Id'])['cost'].count().sort_values(ascending=False)
    df = pd.merge(df1, df2, left_index=True, right_index=True)
    df['utility'] = df['cost_x'] * df['cost_y']
    df['utility'].plot(kind = 'hist', bins=20)
    plt.show()

# in progress
def dura_dist(df: pd.DataFrame, groupby: str = None):
    """Plots distribution of trip durations between 4 most common station pairs.
    Does not modify dataframes.
    <df> is trip data"""    
    if groupby and groupby not in df.columns:
        df = add_col(df, [groupby]) # doesn't modify df
    durations = df.filter(get_label_list(df, ['start station id', 'end station id', 'trip duration min', 'timeperiod']))
    
    # Find 4 most common station pairs
    duration_counts = get_col_count(durations, new_col_name='count', bycol=['start station id', 'end station id'], new=True) #, keep=['trip duration min'])
    check = duration_counts.sort_values('count', ascending=False)
    o = list(check['Start_Station_Id'].iloc[0:4])
    d = list(check['End_Station_Id'].iloc[0:4])

    # Make historgram
    fig, axs = plt.subplots(len(o)//2, 2, figsize=(6, 10)) 
    for i in range(len(o)):
        start, end = o[i], d[i]
        hist_i = durations.loc[lambda durations: (durations['Start_Station_Id'] == start) & (durations['End_Station_Id'] == end)]
        # if groupby:
        #     data = hist_i[['Trip_Duration_min', groupby]]
        #     data.to_csv('test_w_gpt.csv', index=False)
        # else:
        data = hist_i['Trip_Duration_min']
        
        axs[i%2][int(i in [0, 1])].hist(data, bins=30, alpha=0.5, color='b', label=f"{start}-{end}")
        axs[i%2][int(i in [0, 1])].set_title(f'{start} and {end}')
        # axs[i%2][int(i in [0, 1])].set_xlabel('Duration (min)')
        axs[i%2][int(i in [0, 1])].set_ylabel('Frequency')
        # axs[i].legend()
    
    plt.legend()
    plt.suptitle("Durations (mins)") # plt.tight_layout()
    plt.show()

# in progress
def dura_dist_oneplot(df: pd.DataFrame, num_pairs: int = 4, groupby: str = None):
    """Plots ONE layered distribution of trip durations between <num_pairs> most common station pairs.
    Does not modify dataframes.
    If <num_pairs> > 1, will create new windows.
    df must represent trips, and include columns: start id, end id, trip duration"""    
    if groupby and groupby not in df.columns:
        df = add_col(df, [groupby]) # doesn't modify df
    df_cols = get_label_list(df, ['start station id', 'end station id', 'trip duration min', groupby])
    durations = df.filter(df_cols)
    start_col, end_col, trip_dur = df_cols[0], df_cols[1], df_cols[2]
    
    # Find 4 most common station pairs
    duration_counts = get_col_count(durations, new_col_name='count', bycol=['start station id', 'end station id'], new=True) #, keep=['trip duration min'])
    check = duration_counts.sort_values('count', ascending=False)
    o = [7016]# list(check[start_col].iloc[0:num_pairs]) 
    d = [7430]#list(check[end_col].iloc[0:num_pairs])

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
                plt.hist(data_i, bins=bins, density=True, alpha=0.5, label=f"{label}: {mean_} min", color=COLOURS[j])
                j += 1
        else:
            data = hist_i[trip_dur]
            plt.hist(data, bins=30, alpha=0.5, label=f"{start}-{end}")
    
        plt.ylabel('Frequency')
        plt.title(f'Distribution of trip durations (min) {start}-{end}')
        plt.legend()
    plt.show()

# Total trips per hour as a line
# (so convert to “seconds from 12 am?)
# plotted by casual/annual member 
def trips_per_hour(df: pd.DataFrame):
    # add "hour" column which represents the hour (in 24 hour format)
    # regardless of day
    try:
        df['hour'] = df[get_label(df, 'start time')].dt.hour 
    except AttributeError:  # sometimes it doesn't recognize the start time column as a datetime object
        df = add_col(df, ['datetime'])
        df['hour'] = df[get_label(df, 'start time')].dt.hour 
    print(df)
    print(df.columns)

    # split data into two, one for each groupby group
    casual_data = df.loc[df['User_Type'] == 'Casual Member']
    annual_data = df.loc[df['User_Type'] == 'Annual Member']

    # count num of trips per hour, using groupby
    casual_data = get_col_count(casual_data, ['hour'], 'count', new=True)

    annual_data = get_col_count(annual_data, ['hour'], 'count', new=True)
    annual_data = annual_data.merge(right=casual_data['hour'], how='right').fillna(0)
    print(annual_data)

    plt.figure(figsize=(14, 6))
    plt.plot(casual_data['hour'], casual_data['count'], 'b-', annual_data['hour'], annual_data['count'].astype(int), 'g-')
    plt.ylabel('Trip numbers')
    nums = np.arange(0, 24)
    labels = []
    for num in nums:
        if num == 0:
            labels.append('12am') 
        elif num <= 12:
            labels.append(str(num) + 'am')
        else:
            labels.append(str(num - 12) + 'pm')
    plt.xticks(nums, labels=labels)
    plt.xlabel('Hour')
    plt.legend(['Casual', 'Annual'])
    plt.title('Trips taken in 24 hours')
    plt.show()
    return

def minutes_since_midnight(timestamp, interval_size: int = 60) -> float:
    """Returns number of minutes between <timestamp> and midnight
    OLD: If <interval_size> is specified (in minutes), returns the number
    of intervals between <timestamp> and midnight."""
    
    midnight = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    min_since_midnight = (timestamp - midnight).total_seconds() / 60
    # num_interval_per_day = 24 * (60//interval_size)
    # for 60-min intervals: (min_since_midnight//60)*60 
    return (min_since_midnight // interval_size) * interval_size

# OLD
def trips_per_minute(df: pd.DataFrame):
    # add "minute" column which represents the minutes from 12 am
    # regardless of day
    try:
        df['minutes'] = df[get_label(df, 'start time')].apply(minutes_since_midnight)
    
    except TypeError:  # sometimes it doesn't recognize the start time column as a datetime object
        df = add_col(df, ['datetime'])
        df['minutes'] = df[get_label(df, 'start time')].apply(minutes_since_midnight)
    df['minutes'] = df['minutes'].astype(int)

    # split data into two, one for each groupby group
    casual_data = df.loc[df['User_Type'] == 'Casual Member']
    annual_data = df.loc[df['User_Type'] == 'Annual Member']

    # count num of trips per hour, using groupby
    casual_data = get_col_count(casual_data, ['minutes'], 'count', new=True)
    
    plt.figure(figsize=(14, 6))
    if len(annual_data) > 0:
        annual_data = get_col_count(annual_data, ['minutes'], 'count', new=True)
        merger = pd.DataFrame({'minutes': np.arange(0, 1440)})
        annual_data = annual_data.merge(right=merger, how='right', on='minutes').fillna(0)
        plt.plot(casual_data['minutes'], casual_data['count'], 'b-', annual_data['minutes'], annual_data['count'].astype(int), 'g-')
    else:
        plt.plot(casual_data['minutes'], casual_data['count'], 'b-')
    print(annual_data)

    plt.ylabel('Trip numbers')
    labels = []
    nums = np.arange(0, 1440, 60)
    for num in nums:
        if num == 0:
            labels.append('12am') 
        elif num <= 12*60:
            labels.append(str(num//60) + 'am')
        else:
            labels.append(str(num//60 - 12) + 'pm')
    plt.xticks(nums, labels=labels)
    plt.xticks(np.arange(0, 1440, 60))
    plt.xlabel('Hour')
    plt.legend(['Casual', 'Annual'])
    plt.title('Trips taken in 24 hours')
    plt.show()
    pass

def trips_per_interval(df: pd.DataFrame, datetime_col: str, interval_size: int = 60) -> pd.DataFrame: 
    """add "<interval_size>_minute" column which represents the number of
    <interval_size> length intervals from 12 am (regardless of day)
    Note: interval_size default = 60 min (1 hour)
    <df> must have a "start time" column
    ONLY WEEKDAYS"""
    interval_label = f"{interval_size}_minute"
    max_value = 24*60

    # filter for weekdays, then add new colum
    df[interval_label] = df[datetime_col].apply(minutes_since_midnight, args=(interval_size,))
    df[interval_label] = df[interval_label].astype(int)
    grouped_df = get_col_count(df, bycol=[interval_label], new_col_name='count', new=True)
    grouped_df['count'] = grouped_df['count'] / interval_size
    merger = pd.DataFrame({interval_label: np.arange(0, max_value, interval_size)})
    ret = grouped_df.merge(right=merger, how='right', on=interval_label).fillna(0)

    return ret
    
    # pd.DataFrame({interval_label: df[interval_label].astype(int)})

    # OLD
    # # split data into two, one for each groupby group
    # casual_data = df.loc[df['User_Type'] == 'Casual Member']
    # annual_data = df.loc[df['User_Type'] == 'Annual Member']

    # # count num of trips per hour, using groupby
    # casual_data = get_col_count(casual_data, bycol=[interval_label], new_col_name='count', new=True)
    # print(casual_data)
    # plt.figure(figsize=(14, 6))
    # if len(annual_data) > 0: # issue of annual_data having no trips??
    #     annual_data = get_col_count(annual_data, [interval_label], 'count', new=True)
    #     merger = pd.DataFrame({interval_label: np.arange(0, max_value, interval_size)})
    #     annual_data = annual_data.merge(right=merger, how='right', on=interval_label).fillna(0)
    #     plt.plot(casual_data[interval_label], casual_data['count'], 'b-', annual_data[interval_label], annual_data['count'].astype(int), 'g-')
    # else:
    #     print("no data for annual members")
    #     plt.plot(casual_data[interval_label], casual_data['count'], 'b-')

    # plt.ylabel('Trip numbers')
    # labels = []
    # nums = np.arange(0, max_value, 60)
    # for num in nums:
    #     if num == 0:
    #         labels.append('12am') 
    #     elif num <= 12*60:
    #         labels.append(str(num//60) + 'am')
    #     else:
    #         labels.append(str(num//60 - 12) + 'pm')
    # plt.xticks(nums, labels=labels)
    # plt.xlabel('Hour')
    # plt.legend(['Casual', 'Annual'])
    # plt.title(f'Trips taken in 24 hours')
    # plt.show()
    pass

def trips_per_interval_stacker(df: pd.DataFrame, interval_sizes: list[int], weekdays: str = '', extra_label: str = ''):
    """Plots multiple lines from trips_per_interval(), by user type and interval size. 
    <df> must have a "start time" column
    <weekdays> can be 'weekday' or 'weekend'
    """
    # split data into two, one for each groupby group
    max_value = 24*60
    
    datetime_col = get_label(df, 'start time') # ensure valid datetime column
    if df[datetime_col].dtype != 'datetime64[ns]':
        df = add_col(df, ['datetime'])
    
    if weekdays == 'weekday':
    # filter for weekdays only
        df = df.loc[df[datetime_col].dt.weekday < 5]
    elif weekdays == 'weekend':
        df = df.loc[df[datetime_col].dt.weekday > 4]
    else:
        print("Showing all trips, not ignoring weekday or weekend")

    # separate by User Type
    casual_data = df.loc[df['User_Type'] == 'Casual Member']
    annual_data = df.loc[df['User_Type'] == 'Annual Member']
    interval_sizes.sort()
    fig, ax = plt.subplots(figsize=(11, 14))
    for int_size in interval_sizes:
        # interval_label = f"{int_size}_minute"
        casual_i = trips_per_interval(casual_data, datetime_col, int_size)
        x = casual_i[casual_i.columns[0]]
        y = casual_i[casual_i.columns[1]]
        print(x)
        print(y)
        plt.plot(x, y, '-', label=f"Casual, {int_size}-min intervals", alpha=0.5)
        # legend.append(f"Casual, {int_size}-min intervals")
        if len(annual_data) > 1:
            annual_i = trips_per_interval(annual_data, datetime_col, int_size)
            plt.plot(annual_i[annual_i.columns[0]], annual_i[annual_i.columns[1]], '-', label=f"Annual, {int_size}-min intervals")
            # legend.append(f"Annual, {int_size}-min intervals")
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
    plt.title(f'Trips taken in 24 hours ({extra_label}, {weekdays})')

def run_dura_dist_oneplot(trip_data):
    trips_folder = folderProcessor(trip_data)
    df = trips_folder.get_obj().get_df()
    # dfs = trips_folder.get_dfs()
    df = trips_folder.concat_folder()
    # print(df)
    df_casual = df.loc[df['User_Type'] == 'Casual Member']
    df_annual = df.loc[df['User_Type'] == 'Annual Member']
    # print(df_casual)
    # print(df_annual)
    dura_dist_oneplot(df_casual, num_pairs=1, groupby='timeperiod')
    dura_dist_oneplot(df_annual, num_pairs=1, groupby='timeperiod')

if __name__ == '__main__':
    #months = ['08'] # '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    #for month in months:
        #trip_data = folderProcessor(TRIPS, month).get_obj().get_df()
    # trip_data = folderProcessor(TRIPS).concat_folder()
    # trips_per_interval(trip_data, interval_size=10)
        #trips_per_interval_stacker(df=trip_data, interval_sizes=[15], extra_label=month, weekdays='weekday')
    trip_data = folderProcessor(TRIPS, '08').get_obj().get_df()
    dura_dist_oneplot(trip_data)
    # plt.show()



# ///* Start class definition
#  
# Time periods: "Overnight", "AM", "Mid-day", "PM", "Evening"
# ["Freezing", "Cold", "Cool", "Warm"]
# "Winter", "Spring", "Summer", "Fall"
class makeModel():
    """Make a model from a dataframe"""
    
    def __init__(self, df: pd.DataFrame, predict: str, variables: list[str]):
        # print("Getting dummies...")
        # df = pd.get_dummies(df, columns=['User Type', 'Precip', 'Month_x', 'Time Period_start', "Time Period_end", "Temperature Range"], drop_first=True)
        # rename = {}

        # df.rename(rename, axis=1, inplace=True)
        # df['precip'] = df['precip'].astype(int)
        # print(df.columns)
        # new_col = df.groupby('Start_Station_Id', dropna=False).size().reset_index(name='trip_count')
        # df = pd.merge(df, new_col, on='Start_Station_Id')
        # df.reset_index(inplace=True)
        # # FIX ERROR!!!
        self.df = df.copy()
        self.predict = get_label(df, predict)
        self.vars = []
        for var in variables:
            var = get_label(df, var)
            self.vars.append(var)
        self.df.dropna(subset=self.vars + [self.predict])
        self.df.reset_index(inplace=True)
        return

    def make_dot(self, x: str, y: str): 
        x = get_label(self.df, x)
        y = get_label(self.df, y)
        self.df.plot(x = x, y = y, kind = 'scatter')
        plt.show()
    
    def make_lm(self, group: str):
        grouping = get_label(self.df, group)
        formula = make_formula(self.predict, self.vars)
        vc_formula = {}
        model = smf.mixedlm(formula, data=self.df, re_formula='1', groups=grouping)
        result = model.fit()
        print(result.summary())
        return
    
    # def make(self):
    #     formula1 = 'trips ~ Trip_Duration_(min) + Weekday_x + capacity_orig + capacity_dest + User_Type_Casual_Member + precip + Time_Period_start_Evening + Time_Period_start_Mid-day +\
    #                 Time_Period_start_Overnight + Time_Period_start_PM +\
    #                      Time_Period_end_Evening + Time_Period_end_Mid-day +\
    #                         Time_Period_end_Overnight + Time_Period_end_PM +\
    #                              Temperature_Range_Cold + Temperature_Range_Cool +\
    #                                 Temperature_Range_Warm'
    #     formula2 = 'trips ~ Weekday_x + capacity_orig + capacity_dest + User_Type_Casual_Member + precip + Time_Period_start_Evening + Time_Period_start_Midday + Time_Period_start_Overnight + Time_Period_start_PM + Temperature_Range_Cold + Temperature_Range_Cool + Temperature_Range_Warm'
    #     formula3 = 'trip_count ~ Weekday_x + capacity_orig + capacity_dest + User_Type_Casual_Member + precip + Temperature_Range_Cold + Temperature_Range_Cool + Temperature_Range_Warm'
    #     model = smf.mixedlm(formula3, data=df, groups=df['Start_Station_Id'])
    #     result = model.fit()
    #     # # Print the summary of the model
    #     print(result.summary())
    
    
    # def tester(self):
    #     data = {'StudentID': [1, 2, 3, 4, 5, 6, 7], 'SchoolID': [1, 1, 2, 2, 3, 1, 2],
    #             'TestScore': [85, 78, 92, 88, 90, 88, 78], 'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female'],
    #             'SchoolSize': [500, 500, 600, 600, 700, 400, 800], 'SES': ['High', 'High', 'Low', 'Low', 'Medium', 'High', 'High']}
    #     df = pd.DataFrame(data)

    #     formula = 'TestScore ~ Gender + SchoolSize + SES'

    #     # Fit the mixed effects model
    #     mixedlm_model = sm.MixedLM.from_formula(formula, groups='SchoolID', data=df)
    #     result = mixedlm_model.fit()

    #     # Print the summary of the model
    #     print(result.summary())
    
    # def old(self, df: pd.DataFrame):
    #     print("Getting dummies...")
    #     df = pd.get_dummies(df, columns=['User Type', 'Precip', 'Month_x', 'Time Period_start', "Time Period_end", "Temperature Range"], drop_first=True)
    #     rename = {}
    #     for col in df.columns:
    #         rename[col] = col.replace(' ', '_')
    #     rename["Precip_1.0"] = 'precip'

    #     df.rename(rename, axis=1, inplace=True)
    #     df['precip'] = df['precip'].astype(int)
    #     print(df.columns)
    #     new_col = df.groupby('Start_Station_Id', dropna=False).size().reset_index(name='trip_count')
    #     df = pd.merge(df, new_col, on='Start_Station_Id')
    #     df.reset_index(inplace=True)
    #     # FIX ERROR!!!
    #     formula1 = 'trips ~ Trip_Duration_(min) + Weekday_x + capacity_orig + capacity_dest + User_Type_Casual_Member + precip + Time_Period_start_Evening + Time_Period_start_Mid-day +\
    #                 Time_Period_start_Overnight + Time_Period_start_PM +\
    #                      Time_Period_end_Evening + Time_Period_end_Mid-day +\
    #                         Time_Period_end_Overnight + Time_Period_end_PM +\
    #                              Temperature_Range_Cold + Temperature_Range_Cool +\
    #                                 Temperature_Range_Warm'
    #     formula2 = 'trips ~ Weekday_x + capacity_orig + capacity_dest + User_Type_Casual_Member + precip + Time_Period_start_Evening + Time_Period_start_Midday + Time_Period_start_Overnight + Time_Period_start_PM + Temperature_Range_Cold + Temperature_Range_Cool + Temperature_Range_Warm'
    #     formula3 = 'trip_count ~ Weekday_x + capacity_orig + capacity_dest + User_Type_Casual_Member + precip + Temperature_Range_Cold + Temperature_Range_Cool + Temperature_Range_Warm'
    #     model = smf.mixedlm(formula3, data=df, groups=df['Start_Station_Id'])
    #     result = model.fit()
    #     # # Print the summary of the model
    #     print(result.summary())

def make_formula(predict: str, variables: list[str]) -> str:
    formula = predict + ' ~ '
    for i in range(len(variables)):
        ext = ' + '
        if i == len(variables) - 1:
            ext = ''
        formula += variables[i] + ext
    return formula


# more dump
   #st7059_7033 = durations.loc[lambda df: (df['Start_Station_Id'] == 7059) & (df['End_Station_Id'] == 7033)]
    #print(st7059_7033)
    # print(durations.groupby(['Start_Station_Id', 'End_Station_Id'], dropna=False).nunique())
    #st7059_7033['Trip_Duration_min'].plot(kind = 'hist', bins=20, xlabel='Duration (min)')
    #plt.show()

    # PANDAS plot
    # hist_i['Trip_Duration_min'].plot(kind = 'hist', bins=20, xlabel='Duration (min)')

# OLD trips_per_interval
# def trips_per_interval(df: pd.DataFrame, interval_size: int = 60): 
#     """add "<interval_size>_minute" column which represents the number of
#     <interval_size> length intervals from 12 am (regardless of day)
#     Note: interval_size default = 60 min (1 hour)
#     <df> must have a "start time" column
#     ONLY WEEKDAYS"""
#     interval_label = f"{interval_size}_minute"
#     max_value = 24*60
    
#     datetime_col = get_label(df, 'start time') # ensure valid datetime column
#     if df[datetime_col].dtype != 'datetime64[ns]':
#         df = add_col(df, ['datetime'])
    
#     # filter for weekdays, then add new colum
#     df = df.loc[df[datetime_col].dt.weekday < 5]
#     df[interval_label] = df[datetime_col].apply(minutes_since_midnight, args=(interval_size,))
#     df[interval_label] = df[interval_label].astype(int)


    # OLD
    # # split data into two, one for each groupby group
    # casual_data = df.loc[df['User_Type'] == 'Casual Member']
    # annual_data = df.loc[df['User_Type'] == 'Annual Member']

    # # count num of trips per hour, using groupby
    # casual_data = get_col_count(casual_data, bycol=[interval_label], new_col_name='count', new=True)
    # print(casual_data)
    # plt.figure(figsize=(14, 6))
    # if len(annual_data) > 0: # issue of annual_data having no trips??
    #     annual_data = get_col_count(annual_data, [interval_label], 'count', new=True)
    #     merger = pd.DataFrame({interval_label: np.arange(0, max_value, interval_size)})
    #     annual_data = annual_data.merge(right=merger, how='right', on=interval_label).fillna(0)
    #     plt.plot(casual_data[interval_label], casual_data['count'], 'b-', annual_data[interval_label], annual_data['count'].astype(int), 'g-')
    # else:
    #     print("no data for annual members")
    #     plt.plot(casual_data[interval_label], casual_data['count'], 'b-')

    # plt.ylabel('Trip numbers')
    # labels = []
    # nums = np.arange(0, max_value, 60)
    # for num in nums:
    #     if num == 0:
    #         labels.append('12am') 
    #     elif num <= 12*60:
    #         labels.append(str(num//60) + 'am')
    #     else:
    #         labels.append(str(num//60 - 12) + 'pm')
    # plt.xticks(nums, labels=labels)
    # plt.xlabel('Hour')
    # plt.legend(['Casual', 'Annual'])
    # plt.title(f'Trips taken in 24 hours')
    # plt.show()
    