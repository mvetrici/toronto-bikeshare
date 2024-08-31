from pd_helpers import df_from_file, get_count_table
from bike_trips_graphs import make_plot_from_df, trips_per_interval_stacker
import matplotlib.pyplot as plt
from file_interactors import directory_path
from bike_trip_interactors import get_trip_file, get_trip_station

TRIPS = "bikeshare-ridership-2023"
DATA = "other-datasets-2023"
NEW = "nov_trips_stations_subset_wards_income.csv"
NEW = "nov_trips_stations_subset_wards_income.csv"
    

def socio_data(socio_file: str):
    df = df_from_file(directory_path(socio_file))
    df.info()
    print([col for col in df.columns])
    df = get_count_table(df, bycol = ['WARD_NUMBER'], keep = ['total_low_income', 'Population_2022'])
    df['prop_low_inc'] = df['total_low_income']/df['Population_2022']
    make_plot_from_df(df, 'prop_low_inc', 'count', '.')
    df.plot(x = 'prop_low_inc', y = 'count', kind = 'scatter')
    plt.show()


if __name__ == '__main__':

    ods = get_trip_station(TRIPS, DATA, '08', True)
    print(ods)