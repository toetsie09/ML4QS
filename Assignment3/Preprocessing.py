from util.VisualizeDataset import VisualizeDataset
from Chapter3.OutlierDetection import DistributionBasedOutlierDetection
from Chapter3.OutlierDetection import DistanceBasedOutlierDetection
from Chapter3.ImputationMissingValues import ImputationMissingValues
from Chapter3.DataTransformation import LowPassFilter
from MergeDatasets import merge_datasets
from util import util
import sys
import copy
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import os

def chauvenet(dataset):
    """ This function finds outliers based on chauvenet's criterion, outliers are set to NA
        Args:
            dataset: pandas framework
    """
    # DataViz = VisualizeDataset(__file__)
    OutlierDistr = DistributionBasedOutlierDetection()

    # We use Chauvenet's criterion for the final version and apply it to all but the label data...
    for col in [c for c in dataset.columns]:
        # print(f'Measurement is now: {col}')
        dataset = OutlierDistr.chauvenet(dataset, col)
        # Visualize the outliers
        # DataViz.plot_binary_outliers(dataset, col, col + '_outlier')
        dataset.loc[dataset[f'{col}_outlier'] == True, col] = np.nan
        del dataset[col + '_outlier']
    return dataset

def distance(dataset):
    """ This function finds outliers based on distance, outliers are set to NA
        Args:
            dataset: pandas framework
    """
    # DataViz = VisualizeDataset(__file__)
    OutlierDist = DistanceBasedOutlierDetection()

    for col in [c for c in dataset.columns]:
        # print(f"Applying distance outlier criteria for column {col}")
        try:
            dataset = OutlierDist.simple_distance_based(dataset, [col], 'euclidean', 0.10, 0.99)
            # Visualize the outliers
            # DataViz.plot_binary_outliers(dataset, col, 'simple_dist_outlier')
            dataset.loc[dataset['simple_dist_outlier'] == True, col] = np.nan
            del dataset['simple_dist_outlier']

        except MemoryError as e:
            print(
                'Not enough memory available for simple distance-based outlier detection...')
            print('Skipping.')
    return dataset

def mean(dataset):
    """ This function fill in missing values
        Args:
            dataset: a pandas dataframe
    """
    MisVal = ImputationMissingValues()
    for col in [c for c in dataset.columns]:
        MisVal.impute_mean(dataset, col)
    return dataset

def median(dataset):
    """ This function fill in missing values
        Args:
            dataset: a pandas dataframe
    """
    MisVal = ImputationMissingValues()
    for col in [c for c in dataset.columns]:
        MisVal.impute_median(dataset, col)
    return dataset

def interpolate(dataset):
    """ This function fill in missing values
        Args:
            dataset: a pandas dataframe
    """
    MisVal = ImputationMissingValues()
    for col in [c for c in dataset.columns]:
        MisVal.impute_interpolate(dataset, col)
    return dataset

def low_pass_filter(dataset, cutoff=1):
    """ This function applies a low pass filter
        Args:
            dataset: a pandas dataframe
            cutoff: upper limit of low pass filter in Hz
    """
    DataViz = VisualizeDataset(__file__)
    LowPass = LowPassFilter()

    periodic_measurements = ['acc_x', 'acc_y', 'acc_z',
                             'gyro_x', 'gyro_y', 'gyro_z',
                             'magnet_x', 'magnet_y', 'magnet_z']
    milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds / 1000
    fs = float(1000) / milliseconds_per_instance

    for col in periodic_measurements:
        dataset = LowPass.low_pass_filter(dataset, col, fs, cutoff, order=10)
        dataset[col] = dataset[col + '_lowpass']
        # DataViz.plot_dataset(dataset.iloc[int(0.4 * len(dataset.index)):int(0.43 * len(dataset.index)), :],
        #                      [col, col + '_lowpass'], ['exact', 'exact'], ['line', 'line'])

        del dataset[col + '_lowpass']
    return dataset

def preprocess_file(READ_PATH, WRITE_PATH, outlier_method, imputation_method, cutoff=1.5):
    print('Now preprocessing:', READ_PATH)
    dataset = pd.read_csv(READ_PATH, index_col=0)
    dataset.index = pd.to_datetime(dataset.index)
    # print(dataset.head())
    # print('missing values before \n', dataset.isna().sum())
    dataset = outlier_method(dataset)
    # print('missing values after removal\n', dataset.isna().sum())
    dataset = imputation_method(dataset)
    # print('missing values after imputation\n', dataset.isna().sum())
    dataset = low_pass_filter(dataset, cutoff)

    dataset.to_csv(WRITE_PATH)


if __name__ == '__main__':
    # Set up file names and locations.
    READ_PATH = Path('./intermediate_datafiles/raw/')
    WRITE_PATH = Path('./intermediate_datafiles/preprocessed/')

    outlier_method = chauvenet
    imputation_method = interpolate
    filter_cutoff = 1.5

    [path.mkdir(exist_ok=True, parents=True) for path in [WRITE_PATH]]

    for file in os.listdir(READ_PATH):
        preprocess_file(READ_PATH / file, WRITE_PATH / file, outlier_method, imputation_method, filter_cutoff)

    combined_dataset = merge_datasets(WRITE_PATH, dummy=True)
    DataViz = VisualizeDataset(__file__)
    print(outlier_method.__name__)
    title = outlier_method.__name__ + '_' + imputation_method.__name__
    print(title)
    DataViz.plot_dataset_boxplot(combined_dataset, ['acc_x', 'acc_y', 'acc_z'] + ['gyro_x', 'gyro_y', 'gyro_z'] +
                                 ['magnet_x', 'magnet_y', 'magnet_z'], name=f'preprocessed_{title}')

    # Plot all data
    util.print_statistics(combined_dataset)
    DataViz.plot_dataset(combined_dataset, ['acc_', 'gyro_', 'magnet_', 'label'],
                                           ['like', 'like', 'like', 'like'],
                                           ['line', 'line', 'line', 'points'], name=f'preprocessed_{title}')