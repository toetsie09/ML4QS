import pandas as pd

df = pd.read_csv("datasets/crowdsignals/csv-participant-one/labels.csv")
# print(df.head(10))
print(df['label'].unique())
# df2 = pd.read_csv("datasets/crowdsignals/csv-participant-one/labels.csv")
# print(df.head(10))