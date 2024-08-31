from socio_file import socio_interactor
from bike_trip_interactors import get_ods
from bike_trips_graphs import dura_dist_oneplot, run_dura_dist_oneplot
from file_interactors import get_trip_file

TRIPS = "bikeshare-ridership-2023"
DATA = "other-datasets-2023"
WARD = "nov_trips_stations_subset_wards_income.csv"
    
if __name__ == '__main__':
    # socio_interactor(WARD, 'prop_1_2_person')
    # get_ods(TRIPS, DATA, '08')
    trips = get_trip_file(TRIPS, '08')
    dura_dist_oneplot(trips, 1)
    run_dura_dist_oneplot(trips)