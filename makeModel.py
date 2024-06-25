import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt
from pd_helpers import get_label, add_col, get_col_count, get_label_list
from folderProcessor import folderProcessor
from dfObj import dfObj

TRIPS = "bikeshare-ridership-2023"
JANTRIPS = r"Bike share ridership 2023-01.csv"
DATA = 'other-datasets-2023'

def visualize_cost():
    trips_folder = folderProcessor(TRIPS, 'test')
    other_folder = folderProcessor(DATA)
    df = trips_folder.combine_merge(other_folder).get_df()
    df = add_col(df, ['weather', 'cost'])

    df1 = df.groupby(['Start_Station_Id', 'End_Station_Id'])['cost'].mean().sort_values(ascending=False)
    df2 = df.groupby(['Start_Station_Id', 'End_Station_Id'])['cost'].count().sort_values(ascending=False)
    df = pd.merge(df1, df2, left_index=True, right_index=True)
    df['utility'] = df['cost_x'] * df['cost_y']
    # print(df)
    # df = df.loc[lambda df: (df['Start_Station_Id'] == 7059) & (df['End_Station_Id'] == 7033) & (df['User_Type'] == 'Casual Member')]
    #print(df)
    df['utility'].plot(kind = 'hist', bins=20)
    plt.show()
    #print('done')
    # pd.plot()

def dura_dist(df: pd.DataFrame, groupby: str = None):
    """Plots distribution of trip durations between 4 most common station pairs.
    Does not modify dataframes."""    
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
    # OPTION 2 (ONE plot) # plt.figure(figsize=(10, 6))
    for i in range(len(o)):
        start, end = o[i], d[i]
        hist_i = durations.loc[lambda durations: (durations['Start_Station_Id'] == start) & (durations['End_Station_Id'] == end)]
        # OPTION 1
        # hist_i['Trip_Duration_min'].plot(kind = 'hist', bins=20, xlabel='Duration (min)')
        # OPTION 2 (for ONE layered plot)
        # plt.hist(hist_i['Trip_Duration_min'], bins=30, alpha=0.5, label=f"{start}-{end}")
        # OPTION 3 (for multiple plots)
        if groupby:
            data = hist_i[['Trip_Duration_min', groupby]]
            data.to_csv('test_w_gpt.csv', index=False)
        else:
            data = hist_i['Trip_Duration_min']
        
        axs[i%2][int(i in [0, 1])].hist(data, bins=30, alpha=0.5, color='b', label=f"{start}-{end}")
        axs[i%2][int(i in [0, 1])].set_title(f'{start} and {end}')
        # axs[i%2][int(i in [0, 1])].set_xlabel('Duration (min)')
        axs[i%2][int(i in [0, 1])].set_ylabel('Frequency')
        # axs[i].legend()
    
    # CSV
    # data.to_csv('test_w_gpt.csv', index=False)
    
    # OPTION 2
    # plt.xlabel('Duration (min)')
    # plt.ylabel('Frequency')
    # plt.title('Durations')
    plt.legend()
    plt.suptitle("Durations (mins)") # plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    trips_folder = folderProcessor(TRIPS, '09')
    df = trips_folder.get_obj()
    # dfs = trips_folder.get_dfs()
    dura_dist(df.get_df(), 'timeperiod')

    # for df in dfs:
    #     dura_dist(df.get_df())


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