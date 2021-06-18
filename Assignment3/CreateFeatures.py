import os
import pandas as pd
import argparse
import copy

from pathlib import Path
from util.VisualizeDataset import VisualizeDataset
from Chapter4.TemporalAbstraction import NumericalAbstraction
from Chapter4.TemporalAbstraction import CategoricalAbstraction
from Chapter4.FrequencyAbstraction import FourierTransformation
from MergeDatasets import merge_datasets
from util import util

def create_features(READ_PATH, WRITE_PATH, aggregation=True, frequency=True):
    print('Now creating features for:', READ_PATH)
    NumAbs = NumericalAbstraction()
    FreqAbs = FourierTransformation()

    # Read dataset from file
    dataset = pd.read_csv(READ_PATH, index_col=0)
    dataset.index = pd.to_datetime(dataset.index)

    original_columns = [c for c in dataset.columns if not 'label' in c]

    # Compute the number of milliseconds covered by an instance based on the first two rows
    milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds / 1000

    if frequency:
        # print('Creating frequency features')
        fs = float(1000) / milliseconds_per_instance
        ws = int(float(10000) / milliseconds_per_instance)
        for col in original_columns:
            dataset = FreqAbs.abstract_frequency(copy.deepcopy(dataset), [col], ws, fs)
            # Spectral analysis.
            # DataViz.plot_dataset(dataset,
            #                      [col + '_max_freq', col + '_freq_weighted', col + '_pse'],
            #                      ['like', 'like', 'like'], ['line', 'line', 'line'])

    if aggregation:
        # print('Creating aggregation features')
        window_sizes = [int(5 * float(1000) / milliseconds_per_instance)]
        methods = ['mean', 'std']
        for col in [c for c in dataset.columns if not 'label' in c]:
            for ws in window_sizes:
                for method in methods:
                    dataset = NumAbs.abstract_numerical(dataset, [col], ws, method)

            # DataViz.plot_dataset(dataset, [col, col + '_temp_mean', col + '_temp_std'],
            #                      ['exact', 'like', 'like'],
            #                      ['line', 'line', 'line'])


    # The percentage of overlap we allow
    window_overlap = 0.9
    skip_points = int((1 - window_overlap) * ws)
    dataset = dataset.iloc[::skip_points, :]

    dataset.to_csv(WRITE_PATH)

if __name__ == '__main__':
    # Set path variables
    READ_PATH = Path('./intermediate_datafiles/preprocessed/')
    WRITE_PATH = Path('./intermediate_datafiles/features/')

    # Allow creation of new dir
    [path.mkdir(exist_ok=True, parents=True) for path in [WRITE_PATH]]

    for file in os.listdir(READ_PATH):
        create_features(READ_PATH/file, WRITE_PATH/file, frequency=True)

    combined_dataset = merge_datasets(WRITE_PATH, dummy=True)
    DataViz = VisualizeDataset(__file__)
    DataViz.plot_dataset_boxplot(combined_dataset, ['acc_x', 'acc_y', 'acc_z'] + ['gyro_x', 'gyro_y', 'gyro_z'] +
                                 ['magnet_x', 'magnet_y', 'magnet_z'], name=f'preprocessed')

    # Plot all data
    util.print_statistics(combined_dataset)
    DataViz.plot_dataset(combined_dataset, ['acc_', 'gyro_', 'magnet_', 'label'],
                                           ['like', 'like', 'like', 'like'],
                                           ['line', 'line', 'line', 'points'], name=f'preprocessed')
