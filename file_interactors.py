import os, re
from pd_helpers import df_from_file
import pandas as pd

TRIPS = "test-bikeshare-ridership-2023"
DATA = 'other-datasets-2023'

class PathNotFound(Exception):
    def __init__(self, attempted_file):
        """attempted_file can also be a month"""
        self.message = f"Data path to {attempted_file} could not be found"
        super().__init__(self.message)

def check_write(data: pd.DataFrame = pd.DataFrame(), suggested_name: str = 'no suggested name') -> bool:
    checker = input("Are you SURE you want to write this file? (y/n) ")
    if not suggested_name.endswith('.csv'):
        suggested_name += '.csv'
    if re.search(r'y', checker, flags=re.I):
        print("Suggested file name is: " + suggested_name)
        change_name = input("Keep file name ('keep') or change filename (type it out)? ")
        if re.search(r'keep', change_name, flags=re.I):
            file = os.path.join(os.path.dirname(os.getcwd()), suggested_name)
        else:
            file_name = change_name.replace(' ', '_')
            if not file_name.endswith('.csv'):
                file_name += '.csv'
            file = os.path.join(os.path.dirname(os.getcwd()), file_name)
        if not data.empty:
            data.to_csv(file, index=False)
    return bool(re.search(r'y', checker, flags=re.I))
        

def directory_path(file_title: str) -> str:
    """Returns file path so that saved file goes to the directory
    that holds the repo rather than in the repo."""
    return os.path.join(os.path.dirname(os.getcwd()), file_title)

# helper 
def file_path(folder_name: str, file_name: str) -> str:
    print(folder_name)
    print(file_name)
    folder_paths = get_folder_paths(folder_name) # paths in the folder
    data_path = folder_paths.get(file_name)
    print(data_path)
    if data_path: 
        return data_path
    raise PathNotFound(file_name)

# helper 
def get_folder_paths(folder_name: str) -> dict[str, str]:
    folder_path = os.path.abspath(folder_name)
    folder_dict = {}
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            data_path = os.path.join(folder_path, file)
            folder_dict[file] = data_path
    return folder_dict

def get_trip_file(folder_name: str, month: str = '01'):
    """Finds file in folder folder_name. folder_name must be specified.
    file_name is a month specied as 'MM'"""
    file_name = f'Bike share ridership 2023-{month}.csv'
    data_path = file_path(folder_name, file_name)
    
    df = df_from_file(data_path)
    print(f"File '{file_name}' is being processed.")
    return df

def get_file_from_folder(folder_name: str, file_name: str = '', file_type = '') -> pd.DataFrame:
    """Finds file file_name in folder folder_name. Both must be specified.
    Optionally, file_name can refer to the type ('Weather', 'TTCStation', or 'BikeStation')"""
    if file_name:
        data_path = file_path(folder_name, file_name)
    if file_type:
        file_name = file_from_type(folder_name, file_type)
        data_path = file_path(folder_name, file_name)
    return df_from_file(data_path)

# helper
def file_from_type(folder_name: str, file_type: str) -> str:
    """file_type must be ('Weather', 'TTCStation', or 'BikeStation')"""
    for file in get_folder_paths(folder_name).keys():
        if get_file_type(file) == file_type:
            return file
    raise PathNotFound(file_type)

def get_file_type(file_name: str):
    # pure helper (in progress)
    """Assigns types to dataframe objects based on  pathname."""
    if 'weather' in file_name:
        dtype = 'Weather'
    if 'gas' in file_name:
        dtype = 'Gas'
    if 'ttc' in file_name:
        dtype = "TTCStation"
    elif 'stations' in file_name:
        dtype = "BikeStation"
    if dtype:
        return dtype
    else:
        raise PathNotFound("No valid file type")
    