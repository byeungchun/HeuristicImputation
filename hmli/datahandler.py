import os
from pandas import DataFrame, to_datetime, read_hdf, read_csv, to_datetime


def load_bismacro_csv_into_dataframe(file_directory: str, csv_file: str) -> DataFrame:
    """
    Load BIS Macro CSV file into Pandas dataframe
    This CSV file is generated from DBQL and contains all BIS monthly macro series

    :param file_directory:
    :param csv_file: 'bismacromonthly.txt'
    :return: Monthly macro time seires DataFrame
    """
    os.chdir(file_directory)
    lines = list()
    with open(csv_file, 'r') as fin:
        for i, line in enumerate(fin):
            tokens = line.split('\t')
            if len(tokens) < 5:
                continue
            _code = [tokens[1].replace('"', '')]
            _code.extend(tokens[5:])
            lines.append(_code)
    df = DataFrame(lines[1:], columns=lines[0])
    df.index = df.iloc[:, 0]
    df = df.iloc[:, 1:]
    df = df.transpose()
    df.index = to_datetime(df.index)
    df = df.dropna(axis=1, how='all')
    return df


def save_dataframe_to_hdf5(df: DataFrame, hdf5_file: str, df_key: str):
    """
    Save BIS monthly time series dataframe to hdf5 file

    :param df: BIS monthly time series dataframe
    :param hdf5_file: HDF5 file; 'bismacro.h5'
    :param df_key: The dataframe key; 'df'
    """
    df.to_hdf(hdf5_file, df_key)


def load_monthly_time_series_from_hdf5(hdf5_file, df_key='df') -> DataFrame:
    """
    Load BIS monthly time series dataframe from the hdf5 file

    :param hdf5_file: HDF5 file; bismacro.h5
    :param df_key: The dataframe key; 'df'
    :return: BIS monthly time series dataframe
    """
    df = read_hdf(hdf5_file, df_key)
    return df


def convert_oecd_csv_to_hdf5(oecd_csv_file='MEIData-2018-08.csv', hdf5_file='mei.h5', df_key='df'):
    """
    Convert OECD MEI csv file and save it to hdf5 file 
    OECD MEI: https://www.oecd-ilibrary.org/economics/data/main-economic-indicators_mei-data-en
    :param oecd_csv_file:
    :param hdf5_file:
    :param df_key:
    :return:
    """
    df = read_csv(oecd_csv_file)
    mm_cols = [df.columns[0]]  # include series code
    for i in df.columns[10:]:  # take monthly series only
        if i.find('M') > 0:
            mm_cols.append(i)
    df = df[mm_cols]
    df.index = df.iloc[:, 0]
    df = df.iloc[:, -105:-5]  # remove old data because there are many missing values
    df = df.dropna()
    df.columns = to_datetime(df.columns, format='%YM%m')
    df = df.transpose()
    df.to_hdf(hdf5_file, df_key)
    return df
