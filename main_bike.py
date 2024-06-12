# from makeDF import makeDF
from folderProcessor import folderProcessor
# from makeModel import makeModel

bike_path = "bikeshare-ridership-2023"
data_path = "other-datasets-2023"
folder_data= folderProcessor(bike_path, 'test')
folder_data.getinfo()
print(folderProcessor(bike_path, 'test'))

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