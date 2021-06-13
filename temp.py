from pathlib import Path
from Chapter3.OutlierDetection import DistributionBasedOutlierDetection
from Chapter3.ImputationMissingValues import ImputationMissingValues
from Chapter3.DataTransformation import LowPassFilter
from Chapter3.DataTransformation import PrincipalComponentAnalysis
from util.VisualizeDataset import VisualizeDataset
import pandas as pd
import numpy as np
import copy


# Set up file names and locations.
DATA_PATH = Path('./intermediate_datafiles/personal/')
DATASET_FNAME = 'chapter2_result_1000.csv'
RESULT_FNAME = 'chapter3_result_1000.csv'

# Load dataset
try:
    dataset = pd.read_csv(Path(DATA_PATH / DATASET_FNAME), index_col=0)
    dataset.index = pd.to_datetime(dataset.index)

except IOError as e:
    print('File not found, try to run the preceding crowdsignals scripts first!')
    raise e

# We'll create an instance of our visualization class to plot the results.
DataViz = VisualizeDataset(__file__)
OutlierDistr = DistributionBasedOutlierDetection()
MisVal = ImputationMissingValues()
LowPass = LowPassFilter()
PCA = PrincipalComponentAnalysis()

# Chapter 3 outliers
# We use Chauvenet's criterion for the final version and apply it to all but the label data...
for col in [c for c in dataset.columns if not 'label' in c]:
    print(f'Measurement is now: {col}')
    dataset = OutlierDistr.chauvenet(dataset, col)
    dataset.loc[dataset[f'{col}_outlier'] == True, col] = np.nan
    del dataset[col + '_outlier']

# milliseconds_per_instance = (
#     dataset.index[1] - dataset.index[0]).seconds/1000000000


# chapter 3 rest
for col in [c for c in dataset.columns if not 'label' in c]:
    dataset = MisVal.impute_interpolate(dataset, col)

print(dataset.describe())
print('esase')
print(dataset['light_phone_Illuminance'].describe())
print(dataset['press_phone_Millibars'].describe())
# # And now let us include all LOWPASS measurements that have a form of periodicity (and filter them):
# periodic_measurements = ['acc_phone_X', 'acc_phone_Y', 'acc_phone_Z', 'gyr_phone_X', 'gyr_phone_Y',
#                          'gyr_phone_Z']
#
# # Let us apply a lowpass filter and reduce the importance of the data above 1.5 Hz
#
# # Determine the sampling frequency.
# fs = float(1000) / milliseconds_per_instance
# cutoff = 1.5
#
# for col in periodic_measurements:
#     dataset = LowPass.low_pass_filter(
#         dataset, col, fs, cutoff, order=10)
#     dataset[col] = dataset[col + '_lowpass']
#     del dataset[col + '_lowpass']
#
# # We used the optimal found parameter n_pcs = 7, to apply PCA to the final dataset
#
# selected_predictor_cols = [c for c in dataset.columns if (not ('label' in c)) and (not (c == 'hr_watch_rate'))]
#
# n_pcs = 7
#
# dataset = PCA.apply_pca(copy.deepcopy(dataset), selected_predictor_cols, n_pcs)
#
# # # And the overall final dataset:
# # DataViz.plot_dataset(dataset, ['acc_', 'gyr_', 'hr_watch_rate', 'light_phone_lux', 'mag_', 'press_phone_', 'pca_', 'label'],
# #                      ['like', 'like', 'like', 'like', 'like',
# #                          'like', 'like', 'like', 'like'],
# #                      ['line', 'line', 'line', 'line', 'line', 'line', 'line', 'points', 'points'])

# Plot all data
DataViz.plot_dataset(dataset, ['acc_', 'gyr_', 'light_phone_', 'press_phone_', 'label'],
                     ['like', 'like', 'like', 'like', 'like'],
                     ['line', 'line', 'line', 'line', 'points'])

dataset.to_csv(DATA_PATH / RESULT_FNAME)