import os
import pandas as pd

from pathlib import Path

def merge_datasets(READ_PATH, dummy=False):
    """ This function merges all .csv files into one big file and adds labels
        Args:
            READ_PATH: the path to the dir with all files that have to be merged
        Returns:
            final_dataset: a pandas dataframe
    """
    final_dataset = pd.DataFrame()
    index_list = []
    max_timer = None
    for file in os.listdir(READ_PATH):
        temp_dataset = pd.read_csv(READ_PATH/file, index_col=0)
        temp_dataset.index = pd.to_datetime(temp_dataset.index)
        temp_dataset['label'] = 'label' + file[:-6]

        if max_timer == None:
            new_indexes = temp_dataset.index
            max_timer = new_indexes[-1]
        else:
            # Properly adjust the index Timestamps to make create a continuous dataset
            default_timestamp = pd.Timestamp('1970-01-01 00:00:00')
            differences = temp_dataset.index - default_timestamp
            new_indexes = max_timer + differences
        index_list.extend(new_indexes)

        final_dataset = final_dataset.append(temp_dataset, ignore_index=True)

    # Add dummy
    if dummy:
        dummy_features = pd.get_dummies(final_dataset['label'])
        final_dataset = pd.concat([final_dataset, dummy_features], axis=1)
        del final_dataset['label']
    final_dataset.index = index_list

    return final_dataset

if __name__ == "__main__":
    READ_PATH = Path('./intermediate_datafiles/raw/')
    # merge_datasets(READ_PATH)
