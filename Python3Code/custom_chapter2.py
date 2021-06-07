##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 2                                               #
#                                                            #
##############################################################

# Import the relevant classes.
import pandas as pd

from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from util import util
from pathlib import Path
import copy
import os
import sys

# Chapter 2: Initial exploration of the dataset.

DATASET_PATH = Path('./datasets/SensorRecord/FinalFiles')
RESULT_PATH = Path('./intermediate_datafiles/personal')
RESULT_FNAME = 'chapter2_result.csv'

# Set a granularity (the discrete step size of our time series data). We'll use a course-grained granularity of one
# instance per minute, and a fine-grained one with four instances per second.
GRANULARITIES = [60000, 250]

# We can call Path.mkdir(exist_ok=True) to make any required directories if they don't already exist.
[path.mkdir(exist_ok=True, parents=True) for path in [DATASET_PATH, RESULT_PATH]]


datasets = []
for milliseconds_per_instance in GRANULARITIES:
    print(f'Creating numerical datasets from files in {DATASET_PATH} using granularity {milliseconds_per_instance}.')

    # Create an initial dataset object with the base directory for our data and a granularity
    dataset = CreateDataset(DATASET_PATH, milliseconds_per_instance)

    # Add the selected measurements to it.
    #
    # We add the accelerometer data (continuous numerical measurements) of the phone and the smartwatch
    # and aggregate the values per timestep by averaging the values
    dataset.add_numerical_dataset('FinalFiles/Accelerometer.csv', 'Timestamp', ['X','Y', 'Z'], 'avg', 'acc_phone_')
    dataset.add_numerical_dataset('FinalFiles/Gyroscope.csv', 'Timestamp', ['X', 'Y', 'Z'], 'avg', 'gyr_phone_')
    dataset.add_numerical_dataset('FinalFiles/Light.csv', 'Timestamp', ['Illuminance'], 'avg', 'light_phone_')
    dataset.add_numerical_dataset('FinalFiles/Pressure.csv', 'Timestamp', ['Millibars'], 'avg', 'press_phone_')

    dataset.add_event_dataset('FinalFiles/labels.csv', 'label_start_datetime', 'label_end_datetime', 'label', 'binary')

    dataset = dataset.data_table

    # Plot the data
    DataViz = VisualizeDataset(__file__, milliseconds_per_instance)

    print(dataset.columns)

    # Boxplot
    DataViz.plot_dataset_boxplot(dataset, ['acc_phone_X','acc_phone_Y','acc_phone_Z',
                                           'gyr_phone_X', 'gyr_phone_Y', 'gyr_phone_Z'])

    # Plot all data
    DataViz.plot_dataset(dataset, ['acc_', 'gyr_', 'light_phone_', 'press_phone_', 'label'],
                                  ['like', 'like', 'like', 'like', 'like'],
                                  ['line', 'line', 'line', 'points', 'points'])

    # And print a summary of the dataset.
    util.print_statistics(dataset)
    datasets.append(copy.deepcopy(dataset))

    dataset.to_csv(RESULT_PATH / f'chapter2_result_{milliseconds_per_instance}')