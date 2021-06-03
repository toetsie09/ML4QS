import pandas as pd
from pathlib import Path

from Chapter3.ModelBasedImputation import ModelBasedImputation

DATA_PATH = Path('./intermediate_datafiles/')
ORIG_DATASET_FNAME = 'chapter2_result.csv'

# Using the result from Chapter 2, let us try the Kalman filter on the light_phone_lux attribute and study the result.
try:
    original_dataset = pd.read_csv(
        DATA_PATH / ORIG_DATASET_FNAME, index_col=0)
    original_dataset.index = pd.to_datetime(original_dataset.index)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e

modelImpute = ModelBasedImputation()
impute_dataset = modelImpute.apply_model_based_imputation(original_dataset, 'hr_watch_rate', info=True)

# DataViz.plot_imputed_values(kalman_dataset, [
#     'original', 'kalman'], 'acc_phone_x', kalman_dataset['acc_phone_x_kalman'])
# DataViz.plot_dataset(kalman_dataset, ['acc_phone_x', 'acc_phone_x_kalman'], [
#     'exact', 'exact'], ['line', 'line'])