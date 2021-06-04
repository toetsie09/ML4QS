import pandas
import matplotlib
from Python3Code.Chapter3.OutlierDetection import DistributionBasedOutlierDetection


data = pandas.read_csv("../Python3Code/intermediate_datafiles/chapter2_result.csv", index_col=0)
light_phone_col = "light_phone_lux"
acc_phone_col = "acc_phone_x"
