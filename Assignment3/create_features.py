import os
import pandas as pd
import argparse
import copy

from pathlib import Path
from util.VisualizeDataset import VisualizeDataset
from Chapter4.TemporalAbstraction import NumericalAbstraction
from Chapter4.TemporalAbstraction import CategoricalAbstraction
from Chapter4.FrequencyAbstraction import FourierTransformation

def main():
    # Set path variables
    READ_PATH = Path('./intermediate_datafiles/preprocessed/')
    WRITE_PATH = Path('./intermediate_datafiles/features/')

    # Allow creation of new dir
    [path.mkdir(exist_ok=True, parents=True) for path in [WRITE_PATH]]

    # Let us create our visualization class again.
    DataViz = VisualizeDataset(__file__)

    NumAbs = NumericalAbstraction()
    FreqAbs = FourierTransformation()

    # Generate features for all files
    for file in os.listdir(READ_PATH):
        # Read dataset from file
        dataset = pd.read_csv(READ_PATH / file, index_col=0)
        dataset.index = pd.to_datetime(dataset.index)

        original_columns = [c for c in dataset.columns if not 'label' in c]

        # Compute the number of milliseconds covered by an instance based on the first two rows
        milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds / 1000

        if FLAGS.aggregation:
            print('Creating aggregation features')
            window_sizes = [int(5 * float(1000) / milliseconds_per_instance)]
            methods = ['mean', 'std']
            for col in [c for c in dataset.columns if not 'label' in c]:
                for ws in window_sizes:
                    for method in methods:
                        dataset = NumAbs.abstract_numerical(dataset, [col], ws, method)

                DataViz.plot_dataset(dataset, [col, col + '_temp_mean', col + '_temp_std'],
                                     ['exact', 'like', 'like'],
                                     ['line', 'line', 'line'])

        if FLAGS.frequency:
            print('Creating frequency features')

            fs = float(1000) / milliseconds_per_instance
            ws = int(float(10000) / milliseconds_per_instance)
            for col in original_columns:
                dataset = FreqAbs.abstract_frequency(copy.deepcopy(dataset), [col], ws, fs)
                # Spectral analysis.
                DataViz.plot_dataset(dataset,
                                     [col + '_max_freq', col + '_freq_weighted', col + '_pse'],
                                     ['like', 'like', 'like'], ['line', 'line', 'line'])
        # The percentage of overlap we allow
        window_overlap = 0.9
        skip_points = int((1 - window_overlap) * ws)
        dataset = dataset.iloc[::skip_points, :]

        dataset.to_csv(WRITE_PATH / file)

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--aggregation', type=str, default='True', help="New features based on aggregation methods are added to the dataset.")
    parser.add_argument('--frequency', type=str, default='True', help="New features based on frequency methods are added to the dataset.")
    FLAGS, unparsed = parser.parse_known_args()

    main()
