import pandas as pd

file = './datasets/WISDM/WISDM_ar_v1.1_raw.txt'

df = pd.read_csv(file, sep=',', error_bad_lines=False)
