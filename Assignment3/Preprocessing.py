from util.VisualizeDataset import VisualizeDataset
from Chapter3.OutlierDetection import DistributionBasedOutlierDetection
from Chapter3.OutlierDetection import DistanceBasedOutlierDetection
from Chapter3.ImputationMissingValues import ImputationMissingValues
from Chapter3.DataTransformation import LowPassFilter

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
    DataViz = VisualizeDataset(__file__)
    OutlierDistr = DistributionBasedOutlierDetection()

    # We use Chauvenet's criterion for the final version and apply it to all but the label data...
    for col in [c for c in dataset.columns]:
        print(f'Measurement is now: {col}')
        dataset = OutlierDistr.chauvenet(dataset, col)
        # Visualize the outliers
        DataViz.plot_binary_outliers(dataset, col, col + '_outlier')
        dataset.loc[dataset[f'{col}_outlier'] == True, col] = np.nan
        del dataset[col + '_outlier']
    return dataset

def distance(dataset):
    """ This function finds outliers based on distance, outliers are set to NA
        Args:
            dataset: pandas framework
    """
    DataViz = VisualizeDataset(__file__)
    OutlierDist = DistanceBasedOutlierDetection()

    for col in [c for c in dataset.columns]:
        print(f"Applying distance outlier criteria for column {col}")
        try:
            dataset = OutlierDist.simple_distance_based(dataset, [col], 'euclidean', FLAGS.dmin, FLAGS.fmin)
            # Visualize the outliers
            # DataViz.plot_binary_outliers(dataset, col, 'simple_dist_outlier')
            dataset.loc[dataset['simple_dist_outlier'] == True, col] = np.nan
            del dataset['simple_dist_outlier']

        except MemoryError as e:
            print(
                'Not enough memory available for simple distance-based outlier detection...')
            print('Skipping.')
    return dataset

def imputation_mean(dataset):
    """ This function fill in missing values
        Args:
            dataset: a pandas dataframe
    """
    MisVal = ImputationMissingValues()
    for col in [c for c in dataset.columns]:
        MisVal.impute_mean(dataset, col)
    return dataset

def imputation_median(dataset):
    """ This function fill in missing values
        Args:
            dataset: a pandas dataframe
    """
    MisVal = ImputationMissingValues()
    for col in [c for c in dataset.columns]:
        MisVal.impute_median(dataset, col)
    return dataset

def imputation_interpolate(dataset):
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
                             'gyro_x', 'gyro_y', 'gyro_z']
    milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds / 1000
    fs = float(1000) / milliseconds_per_instance

    for col in periodic_measurements:
        dataset = LowPass.low_pass_filter(dataset, col, fs, cutoff, order=10)
        dataset[col] = dataset[col + '_lowpass']
        # DataViz.plot_dataset(dataset.iloc[int(0.4 * len(dataset.index)):int(0.43 * len(dataset.index)), :],
        #                      [col, col + '_lowpass'], ['exact', 'exact'], ['line', 'line'])

        del dataset[col + '_lowpass']
    return dataset

# Set up file names and locations.
READ_PATH = Path('./intermediate_datafiles/raw/')
WRITE_PATH = Path('./intermediate_datafiles/preprocessed/')

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main():
    print_flags()

    outlier_method = distance
    imputation_method = imputation_interpolate

    [path.mkdir(exist_ok=True, parents=True) for path in [WRITE_PATH]]

    files = os.listdir(READ_PATH)
    print('all listed files in dir', files)
    for file in files:
        dataset = pd.read_csv(READ_PATH / file, index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
        print(dataset.head())
        print('missing values before \n', dataset.isna().sum())
        dataset = outlier_method(dataset)
        print('missing values after removal\n', dataset.isna().sum())
        dataset = imputation_method(dataset)
        print('missing values after imputation\n', dataset.isna().sum())
        dataset = low_pass_filter(dataset)

        DataViz = VisualizeDataset(__file__)

        # Boxplot
        DataViz.plot_dataset_boxplot(dataset, ['acc_x', 'acc_y', 'acc_z'] + ['gyro_x', 'gyro_y', 'gyro_z'])

        # Plot all data
        DataViz.plot_dataset(dataset, ['acc_', 'gyro'],
                                      ['like', 'like'],
                                      ['line', 'line'])

        dataset.to_csv(WRITE_PATH / file)


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='final',
                        help="Select what version to run: LOF, distance, mixture, chauvenet or final \
                        'LOF' applies the Local Outlier Factor to a single variable \
                        'distance' applies a distance based outlier detection method to a single variable \
                        'mixture' applies a mixture model to detect outliers for a single variable\
                        'chauvenet' applies Chauvenet outlier detection method to a single variable \
                        'final' is used for the next chapter",
                        choices=['LOF', 'distance', 'mixture', 'chauvenet', 'final'])

    parser.add_argument('--K', type=int, default=5,
                        help="Local Outlier Factor:  K is the number of neighboring points considered")

    parser.add_argument('--dmin', type=int, default=0.10,
                        help="Simple distance based:  dmin is ... ")

    parser.add_argument('--fmin', type=int, default=0.99,
                        help="Simple distance based:  fmin is ... ")

    FLAGS, unparsed = parser.parse_known_args()

    main()