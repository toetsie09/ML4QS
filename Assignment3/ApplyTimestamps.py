import datetime
import pandas as pd

from pathlib import Path
from CustomCreateDataset import CreateDataset
from MergeDatasets import merge_datasets
from util.VisualizeDataset import VisualizeDataset
from util import util

def collect_dir_paths(root_dir):
    """ This function collects all subfolders in specified folder.
    Args:
        root_dir: root folder in which the function will look for subfolders
    Returns:
        dirs_paths: a list with all paths
        dirs_names: a list with all folder names (to easily ID the type of activity)
    """
    dirs_paths = []
    dirs_names = []
    for path in Path(root_dir).iterdir():
        if path.is_dir() and path != root_dir / 'FinalFiles':
            dirs_paths.append(path)
            dirs_names.append(path.name)
    return dirs_paths, dirs_names

def remove_edges(dataset, cutoff):
    """ This function removes the first and last seconds from the dataset to account for taking the phone in and out of
        the pocket.
    Args:
        dataset: pandas frame
        cutoff: number of seconds that have to be cut
    Returns:
        dataset: a panda frame without the edges
    """
    # print('min', min(dataset.index))
    # print('max', max(dataset.index))
    new_start = min(dataset.index) + datetime.timedelta(seconds=cutoff)
    new_end = max(dataset.index) - datetime.timedelta(seconds=cutoff)
    dataset = dataset[(dataset.index >= new_start) & (dataset.index <= new_end)]
    return dataset

def process_single_exercise(DATA_PATH, DATA_NAME, RESULT_PATH, cutoff=0, granularity=250):
    """ processes the raw data of a single exercise
    Args:
        filename
        cut-off: the number of seconds that have to be cut-off
        granularity: the value for delta t

    Returns:
        Returns nothing, but saves the pandas as a csv file
    """
    print(DATA_PATH)
    dataset = CreateDataset(DATA_PATH, granularity)

    acc_names = ['X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)']
    acc_renames = ['acc_x', 'acc_y', 'acc_z']
    dataset.add_numerical_dataset('Accelerometer.csv', 'Time (s)', acc_names, 'avg', '')

    gyro_names = ['X (rad/s)', 'Y (rad/s)', 'Z (rad/s)']
    gyro_renames = ['gyro_x', 'gyro_y', 'gyro_z']
    dataset.add_numerical_dataset('Gyroscope.csv', 'Time (s)', gyro_names, 'avg', '')

    magnet_names = ['X (µT)', 'Y (µT)', 'Z (µT)']
    magnet_renames = ['magnet_x', 'magnet_y', 'magnet_z']
    dataset.add_numerical_dataset('Magnetometer.csv', 'Time (s)', magnet_names, 'avg', '')

    # Get the resulting pandas data table
    dataset = dataset.data_table

    # print('type', dataset.index.dtype)

    # Rename the inconvenient column names
    dataset.columns = acc_renames + gyro_renames + magnet_renames

    # Show the number of missing values per column
    # print(dataset.shape)
    # print(dataset.isna().sum())

    if cutoff > 0:
        dataset = remove_edges(dataset, cutoff)

    dataset.to_csv(RESULT_PATH / f'{DATA_NAME}.csv')


if __name__ == '__main__':
    # number of seconds to remove near edges
    cutoff = 5
    # level of granularity
    granularity = 250

    ROOT_DIR = Path('datasets/')
    RESULT_PATH = Path('./intermediate_datafiles/raw/')
    DIRS_PATHS, DIRS_NAMES = collect_dir_paths(ROOT_DIR)
    # print(DIRS_PATHS)
    # print(DIRS_NAMES)

    # We can call Path.mkdir(exist_ok=True) to make any required directories if they don't already exist.
    [path.mkdir(exist_ok=True, parents=True) for path in [ROOT_DIR, RESULT_PATH]]

    # Process a single exercise
    for DIR, NAME in zip(DIRS_PATHS, DIRS_NAMES):
        process_single_exercise(DIR, NAME, RESULT_PATH, cutoff=cutoff, granularity=granularity)

    combined_dataset = merge_datasets(RESULT_PATH, dummy=True)
    DataViz = VisualizeDataset(__file__)
    DataViz.plot_dataset_boxplot(combined_dataset, ['acc_x', 'acc_y', 'acc_z'] + ['gyro_x', 'gyro_y', 'gyro_z'] +
                                 ['magnet_x', 'magnet_y', 'magnet_z'], name=f'raw_{granularity}_{cutoff}')

    # Plot all data
    # util.print_statistics(combined_dataset)
    DataViz.plot_dataset(combined_dataset, ['acc_', 'gyro_', 'magnet_', 'label'],
                                           ['like', 'like', 'like', 'like'],
                                           ['line', 'line', 'line', 'points'], name=f'raw_{granularity}_{cutoff}')