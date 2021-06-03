from pathlib import Path
import pandas as pd

def collect_dir_paths(root_dir):
    dirs_paths = []
    dirs_names = []
    for path in Path(root_dir).iterdir():
        if path.is_dir() and path != root_dir / 'FinalFiles':
            dirs_paths.append(path)
            dirs_names.append(path.name)
    return dirs_paths, dirs_names

def combine_measurement_files(dir_paths, dir_names, result_path):
    attributes = ['Accelerometer.csv', 'Gyroscope.csv', 'Light.csv', 'Pressure.csv']

    df_labels = pd.DataFrame(columns=['label', 'label_start_datetime', 'label_end_datetime'])
    df_list = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]

    for i, d in enumerate(dir_paths):
        max_time = None
        min_time = None

        for j, a in enumerate(attributes):
            file_path = d / a
            # print(file_path)
            temp_df = pd.read_csv(file_path)
            # print('file len', len(temp_df))

            # Select the beginning and ending time stamps
            if min_time is None or min(temp_df["Timestamp"]) < min_time:
                min_time = min(temp_df["Timestamp"])

            if max_time is None or max(temp_df["Timestamp"]) > max_time:
                max_time = max(temp_df["Timestamp"])

            # Add the measurement into the collective dataframe
            df_list[j] = pd.concat([df_list[j], temp_df])

        # Assign the label to the according timepoints
        df_labels = df_labels.append({'label': dir_names[i][0:len(dir_names[i])-2],
                                      'label_start_datetime': min_time,
                                      'label_end_datetime': max_time}, ignore_index=True)
    df_labels.to_csv(result_path/'labels.csv')

    # Save the combined attribute files into a csv file
    for i, a in enumerate(attributes):
        df_list[i].to_csv(result_path/a)

ROOT_DIR = Path('./SensorRecord')
DIRS_PATHS, DIRS_NAMES = collect_dir_paths(ROOT_DIR)
RESULT_PATH = Path('./SensorRecord/FinalFiles')

combine_measurement_files(DIRS_PATHS, DIRS_NAMES, RESULT_PATH)