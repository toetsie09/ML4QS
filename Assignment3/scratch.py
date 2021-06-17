import datetime

import pandas as pd

df = pd.read_csv("datasets/test_0/Gyroscope.csv")
print(df['Time (s)'])
df['Time (s)'] = pd.to_datetime(df['Time (s)'], unit='s')
print(df['Time (s)'])

print('min', min(df['Time (s)']))
print('max', max(df['Time (s)']))

print(pd.date_range(min(df['Time (s)']), max(df['Time (s)']), freq=str(250)+'ms'))

# tdelta = pd.to_timedelta(df["Milliseconds"]%1000, unit="ms")
# df['Timestamp'] = pd.to_datetime(df['Timestamp'])
# df["Time1"] = df["Timestamp"] + tdelta

# # print(df['Time1'].tail())
# # print(max(df['Time1']))
# print(df.columns)
# print(df['Timestamp'])