from itertools import repeat

import pandas
import copy
import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis as kurt

from Python3Code.util.VisualizeDataset import VisualizeDataset
from Python3Code.Chapter4.FrequencyAbstraction import FourierTransformation

visualizer = VisualizeDataset()
FreqAbs = FourierTransformation()
freqs = None
temp_list = []

def find_fft_transformation(data):
    # Create the transformation, this includes the amplitudes of both the real
    # and imaginary part.
    global freqs
    global temp_list
    # Estimate power spectral density
    transformation = np.fft.rfft(data, len(data))
    # real
    real_ampl = transformation.real
    # max
    max_freq = freqs[np.argmax(real_ampl[0:len(real_ampl)])]
    # weigthed
    freq_weigthed = float(np.sum(freqs * real_ampl)) / np.sum(real_ampl)

    # pse
    PSD = np.divide(np.square(real_ampl), float(len(real_ampl)))
    PSD_pdf = np.divide(PSD, np.sum(PSD))

    # *** ADDDITIONAL METRICS
    skewness = float(skew(real_ampl))
    kurtosis = float(kurt(real_ampl))
    sampled_freqs, powers_per_sampled_freq = signal.periodogram(real_ampl, fs=freq_weigthed)
    max_estim_power_spect_density = max(powers_per_sampled_freq)

    # Make sure there are no zeros.
    if np.count_nonzero(PSD_pdf) == PSD_pdf.size:
        pse = -np.sum(np.log(PSD_pdf) * PSD_pdf)
    else:
        pse = 0

    real_ampl = np.insert(real_ampl, 0, max_freq)
    real_ampl = np.insert(real_ampl, 0, freq_weigthed)
    real_ampl = np.insert(real_ampl, 0, skewness)
    real_ampl = np.insert(real_ampl, 0, kurtosis)
    real_ampl = np.insert(real_ampl, 0, max_estim_power_spect_density)

    row = np.insert(real_ampl, 0, pse)
    temp_list.append(row)
    return 0


# Get frequencies over a certain window.
def abstract_frequency(data_table, columns, window_size, sampling_rate):
    global freqs
    global temp_list
    freqs = (sampling_rate * np.fft.rfftfreq(int(window_size))).round(3)
    for col in columns:
        collist = []
        # prepare column names
        collist.append(col + '_max_freq')
        collist.append(col + '_freq_weighted')
        collist.append(col + '_pse')
        collist.append(col + "_skewness")
        collist.append(col + "_kurtosis")
        collist.append(col + "_max_estim_power_spect_density")

        collist = collist + [col + '_freq_' +
                             str(freq) + '_Hz_ws_' + str(window_size) for freq in freqs]

        # rolling statistics to calculate frequencies, per window size.
        # Pandas Rolling method can only return one aggregation value.
        # Therefore values are not returned but stored in temp class variable 'temp_list'.

        # note to self! Rolling window_size would be nicer and more logical! In older version windowsize is actually 41. (ws + 1)
        data_table[col].rolling(
            window_size + 1).apply(find_fft_transformation)

        # Pad the missing rows with nans
        frequencies = np.pad(np.array(temp_list), ((40, 0), (0, 0)),
                             'constant', constant_values=np.nan)

        # add new freq columns to frame
        data_table[collist] = pandas.DataFrame(frequencies, index=data_table.index)

        # reset temp-storage array
        del temp_list[:]
    return data_table


if __name__ == '__main__':
    # Load data (use ch3 csv data)
    dataset = pandas.read_csv("../Python3Code/intermediate_datafiles/chapter3_result_outliers.csv", index_col=0)
    dataset.index = pandas.to_datetime(dataset.index)

    # Chose this because it looked more stable
    col = 'acc_phone_x'
    # Columns we are interested in
    cols = ['acc_phone_x_max_freq', 'acc_phone_x_freq_weighted', 'acc_phone_x_pse', 'label']

    milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds / 1000
    fs = float(1000) / milliseconds_per_instance
    ws = int(float(10000) / milliseconds_per_instance)

    data_abstract_f = abstract_frequency(copy.deepcopy(dataset), [col], ws, fs)

    visualizer.plot_dataset(data_abstract_f,
                            ['acc_phone_x_max_freq',
                              'acc_phone_x_freq_weighted',
                              'acc_phone_x_pse',
                              'acc_phone_x_skewness',
                              'acc_phone_x_kurtosis',
                              'acc_phone_x_max_estim_power_spect_density',
                              'label'],
                            list(repeat('like', times=7)),
                            list(repeat('line', times=6)) + ['points'])
