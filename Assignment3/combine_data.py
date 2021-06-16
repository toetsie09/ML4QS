from pathlib import Path
import pandas as pd
from CustomCreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
import datetime

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

    acc_names = ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)']
    acc_renames = ['acc_x', 'acc_y', 'acc_z']
    dataset.add_numerical_dataset('Accelerometer.csv', 'Time (s)', acc_names, 'avg', '')

    gyro_names = ['Gyroscope x (rad/s)', 'Gyroscope y (rad/s)', 'Gyroscope z (rad/s)']
    gyro_renames = ['gyro_x', 'gyro_y', 'gyro_z']
    dataset.add_numerical_dataset('Gyroscope.csv', 'Time (s)', gyro_names, 'avg', '')
    # dataset.add_numerical_dataset('RotationVector.csv', 'Timestamp', ['X', 'Y', 'Z', 'cos', 'headingAccuracy'], 'avg', 'rot_phone_')

    # Get the resulting pandas data table
    dataset = dataset.data_table

    print('type', dataset.index.dtype)

    # Rename the inconvenient column names
    dataset.columns = acc_renames + gyro_renames
    print(dataset.head())

    # Show the number of missing values per column
    print(dataset.shape)
    print(dataset.isna().sum())

    # Plot the data
    DataViz = VisualizeDataset(__file__)

    # Boxplot
    DataViz.plot_dataset_boxplot(dataset, acc_renames+gyro_renames)

    # Plot all data
    DataViz.plot_dataset(dataset, ['acc_', 'gyro'],
                                  ['like', 'like'],
                                  ['line', 'line'])

    dataset = remove_edges(dataset, cutoff)

    dataset.to_csv(RESULT_PATH / f'{DATA_NAME}.csv')

# def combine_measurement_files(dir_paths, dir_names, result_path):
#     attributes = ['Accelerometer.csv', 'Gyroscope.csv', 'Light.csv', 'Pressure.csv']
#
#     df_labels = pd.DataFrame(columns=['label', 'label_start_datetime', 'label_end_datetime'])
#     df_list = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
#
#     for i, d in enumerate(dir_paths):
#         max_time = None
#         min_time = None
#
#         for j, a in enumerate(attributes):
#             file_path = d / a
#             # print(file_path)
#             temp_df = pd.read_csv(file_path)
#             # print('file len', len(temp_df))
#
#             # Select the beginning and ending time stamps
#             if min_time is None or min(temp_df["Timestamp"]) < min_time:
#                 min_time = min(temp_df["Timestamp"])
#
#             if max_time is None or max(temp_df["Timestamp"]) > max_time:
#                 max_time = max(temp_df["Timestamp"])
#
#             # Add the measurement into the collective dataframe
#             df_list[j] = pd.concat([df_list[j], temp_df])
#
#         # Assign the label to the according timepoints
#         df_labels = df_labels.append({'label': dir_names[i][0:len(dir_names[i])-2],
#                                       'label_start_datetime': min_time,
#                                       'label_end_datetime': max_time}, ignore_index=True)
#     df_labels.to_csv(result_path/'labels.csv')
#
#     # Save the combined attribute files into a csv file
#     for i, a in enumerate(attributes):
#         df_list[i].to_csv(result_path/a)

if __name__ == '__main__':
    # number of seconds to remove near edges
    cutoff = 5
    # level of granularity
    granularity = 1000

    ROOT_DIR = Path('datasets/')
    RESULT_PATH = Path('./intermediate_datafiles/raw/')
    DIRS_PATHS, DIRS_NAMES = collect_dir_paths(ROOT_DIR)
    # print(DIRS_PATHS)
    # print(DIRS_NAMES)

    # We can call Path.mkdir(exist_ok=True) to make any required directories if they don't already exist.
    [path.mkdir(exist_ok=True, parents=True) for path in [ROOT_DIR, RESULT_PATH]]

    # Process a single exercise
    # for DIR, NAME in zip(DIRS_PATHS, DIRS_NAMES):
    process_single_exercise(DIRS_PATHS[0], DIRS_NAMES[0], RESULT_PATH)
