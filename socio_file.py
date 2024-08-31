from file_interactors import directory_path
from pd_helpers import df_from_file, get_count_table
import matplotlib.pyplot as plt
import pandas as pd

def modify_socio(socio_file: str) -> pd.DataFrame:
    df = df_from_file(directory_path(socio_file))
    df['prop_low_inc'] = df['total_low_income']/df['Population_2022']
    df['prop_young_adult'] = (df['20_to_24_years'] + df['25_to_29_years']) \
        / df['Population_2022']
    df['prop_rented'] = df['housing_rented'] / (df['housing_rented'] + df['housing_owned'])
    df['prop_bach'] = df['diploma_bach_or_higher'] / df['total_diploma']
    df['prop_1_2_person'] = (df['hh_1_person'] + df['hh_2_person']) / df['total_hhsize']
    return df

def ward_trend(df: pd.DataFrame, x: str):
    df = get_count_table(df, bycol = ['WARD_NUMBER'], keep = [x])
    df.plot(x = x, y = 'count', kind = 'scatter')
    plt.show()

def socio_interactor(socio_file: str, x: str):
    df = modify_socio(socio_file)
    ward_trend(df, x)
