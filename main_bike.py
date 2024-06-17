from folderProcessor import folderProcessor, df_from_file
from makeModel import makeModel

OD_TEST = r"C:\Users\mvetr\OneDrive - University of Toronto\Desktop\Summer research\bike_model_06_24\Bike share ridership 2023-08.csvMERGEstations_data.csv"

bike_path = "bikeshare-ridership-2023"
data_path = "other-datasets-2023"
bike_data = folderProcessor(bike_path, 'test')
add_data = folderProcessor(data_path)
# objs = bike_data.get_ods(add_data)
testdf = df_from_file(OD_TEST)
makeModel(df=testdf, predict='count_orig', vars = ['capacity', 'count_dest']).make_dot(x='capacity', y='dest count')




# folder_data.getinfo()
# print(folderProcessor(bike_path, 'test'))


# bikeshare = makeDF(bike_path, data_path, types=['Bike Station'], test='test')
# df = bikeshare.get_df()
# print(df)
# df.info()
# print(df.columns)

# NEXT: do makemodel after figuring out which columns are meaningful 
# after grouping by start station id

# model_test = makeModel(df)
# print(bikeshare)
# bikeshare.getinfo()

# datasets = folderProcessor(data_path)
# datasets.getinfo()


# TODO: make dataframes into subclasses?? so can give month title? or leave for now
# finish add_col function DONE!
# finish implementing mergeDF subclass DONE!
# test columns NOT DONE
# implement merge od NOT DONE
# try model 
# need to implement gas by itself