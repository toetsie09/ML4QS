import pandas

dat = pandas.read_csv("chapter3_result_final.csv")
print(dat)
print(dat.columns)
print(dat["Unnamed: 0"])
dat = dat.rename(columns={"Unnamed: 0": "time"})
dat = dat.sort_values("time")
dat["label"] = "Name"

dat.loc[dat["labelWalking"] == 1, "label"] = "Walking"
dat.loc[dat["labelCycling"] == 1, "label"] = "Cycling"
dat.loc[dat["labelOnTable"] == 1, "label"] = "OnTable"
dat.loc[dat["labelScreenTime"] == 1, "label"] = "ScreenTime"
dat.loc[dat["labelSitting"] == 1, "label"] = "Sitting"
dat.loc[dat["labelYardwork"] == 1, "label"] = "Yardwork"

dat = dat.drop(["time", "labelWalking", "labelCycling", "labelOnTable", "labelScreenTime", "labelSitting", "labelYardwork"], axis=1)

print(dat)
print(dat.columns)

dat.to_csv("final_v2.csv")