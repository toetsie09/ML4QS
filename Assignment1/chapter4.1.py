import pandas
import copy

from util.VisualizeDataset import VisualizeDataset
from Chapter4.FrequencyAbstraction import FourierTransformation

visualizer = VisualizeDataset()
FreqAbs = FourierTransformation()

def get_statistics(data, cols):
    for col in cols:
        if col != 'label':
            print ('\n\n')
            print ('Feature: ',col)
            print ('mean: ', data[col].mean())
            print ('std: ', data[col].std())
            print ('max: ', data[col].max())
            print ('min: ',data[col].min())


def extract_frequencies(dataset, columns, label):
    # Let's retrieve same length as running for better comparison
    print(label)
    print(len(dataset[dataset[label] != 0]))
    activity = dataset[dataset[label] != 0][:len(dataset[dataset["labelRunning"] != 0])]
    # Visualize data
    visualizer.plot_dataset(activity, cols, ['like', 'like', 'like', 'like'], ['line', 'line', 'line', 'line'])
    # Get statistics of relevant columns
    get_statistics(activity, columns)


if __name__ == '__main__':
    # Load data (use ch3 csv data)
    data = pandas.read_csv("../intermediate_datafiles/chapter3_result_outliers.csv", index_col=0)
    data.index = pandas.to_datetime(data.index)

    # Chose this because it looked more stable
    col = 'acc_phone_x'

    # Columns we are interested in
    cols = ['acc_phone_x_max_freq', 'acc_phone_x_freq_weighted', 'acc_phone_x_pse']

    milliseconds_per_instance = (data.index[1] - data.index[0]).microseconds / 1000
    fs = float(1000) / milliseconds_per_instance
    ws = int(float(10000) / milliseconds_per_instance)

    # Do Fast Fourier Transform
    # Calculate new features: max_freq, freq_weighted, power spectral entropy
    data_abstract_f = FreqAbs.abstract_frequency(copy.deepcopy(data), [col], ws, fs)

    extract_frequencies(data_abstract_f, cols, "labelWalking")
    extract_frequencies(data_abstract_f, cols, "labelRunning")
