from folderProcessor import folderProcessor
from dfObj import dfObj
from pd_helpers import df_from_file, add_col
import pandas as pd

bike_path = "bikeshare-ridership-2023"
data_path = "other-datasets-2023"
bike_data = folderProcessor(bike_path, 'test')
add_data = folderProcessor(data_path)
obj = bike_data.combine_merge(add_data)
df = obj.get_df()
df['day_count'] = df.groupby(['date', 'Start_Station_Id'])['Start_Station_Id'].transform('count')

df = add_col(df, ['timeperiod'])
df = pd.get_dummies(df, columns=["timeperiod_start"], drop_first=True)
print(df)
dfObj('test-with-groupby', df).write()
print(df)

# print(obj)



#  obj.getinfo(na=True)
variables = ['trip_count', 'capacity_orig', 'user type', 'max temp c']
# model = makeModel(obj.get_df(), predict=variables[0], variables=variables[1:])
# model.make_lm('date')

 # makeModel(df=obj.get_df(), predict='count_orig', vars = ['capacity', 'count_dest']).make_dot(x='capacity', y='dest count')
# OD_TEST
# testdf = df_from_file(OD_TEST)
# makeModel(df=testdf, predict='count_orig', vars = ['capacity', 'count_dest']).make_dot(x='capacity', y='dest count')
