from folderProcessor import folderProcessor, df_from_file
from makeModel import makeModel

bike_path = "bikeshare-ridership-2023"
data_path = "other-datasets-2023"
bike_data = folderProcessor(bike_path, 'test')
add_data = folderProcessor(data_path)
objs = bike_data.get_ods(add_data)
for obj in objs:
    print(obj)
    obj.write()
    # makeModel(df=obj.get_df(), predict='count_orig', vars = ['capacity', 'count_dest']).make_dot(x='capacity', y='dest count')
# OD_TEST
# testdf = df_from_file(OD_TEST)
# makeModel(df=testdf, predict='count_orig', vars = ['capacity', 'count_dest']).make_dot(x='capacity', y='dest count')


# model_test = makeModel(df)
# print(bikeshare)
# bikeshare.getinfo()

