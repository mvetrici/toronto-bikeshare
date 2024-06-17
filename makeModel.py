import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
from dfObj import find_groupby
import matplotlib.pyplot as plt


# Time periods: "Overnight", "AM", "Mid-day", "PM", "Evening"
# ["Freezing", "Cold", "Cool", "Warm"]
# "Winter", "Spring", "Summer", "Fall"
class makeModel():
    """Make a model from a dataframe"""
    
    def __init__(self, df: pd.DataFrame, predict: str, vars: list[str]):
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
        self.predict = find_groupby(df, predict)
        self.new_vars = []
        for var in vars:
            var = find_groupby(df, var)
            self.new_vars.append(var)
        return

    def make_dot(self, x: str, y: str): 
        x = find_groupby(self.df, x)
        y = find_groupby(self.df, y)
        self.df.plot(x = x, y = y, kind = 'scatter')
        plt.show()
    
    def make_lm(self):
        formula = make_formula(self.predict, self.new_vars)
        model = smf.mixedlm(self, data=self.df, groups=self.df.columns[0])
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