import math
from typing import List

import numpy
import pandas
import copy
from scipy import special
from sklearn.mixture import GaussianMixture

from Python3Code.Chapter3.OutlierDetection import DistanceBasedOutlierDetection
from Python3Code.util.VisualizeDataset import VisualizeDataset


visualizer = VisualizeDataset()
outlier_techniques = DistanceBasedOutlierDetection()


def chauvenet(data_table: pandas.DataFrame, col: str, c: float) -> pandas.DataFrame:
    # Computer the mean and standard deviation.
    mean = data_table[col].mean()
    std = data_table[col].std()
    N = len(data_table.index)
    criterion = 1.0 / (c * N)

    # Consider the deviation for the data points.
    deviation = abs(data_table[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(2)
    high = deviation / math.sqrt(2)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(data_table.index)):
        # Determine the probability of observing the point
        prob.append(1.0 - 0.5 * (special.erf(high[i]) - special.erf(low[i])))
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    data_table[col + '_outlier'] = mask
    return data_table


def mixture_model(data_table: pandas.DataFrame, col: str, k: float) -> pandas.DataFrame:
    # Fit a mixture model to our data.
    dataset = data_table[data_table[col].notnull()][col]
    g = GaussianMixture(n_components=k, max_iter=100, n_init=1)
    reshaped_data = numpy.nan_to_num(numpy.array(data.values.reshape(-1, 1)))
    g.fit(reshaped_data)

    # Predict the probabilities
    probs = g.score(reshaped_data)

    # Create the right data frame and concatenate the two.
    data_probs = pandas.DataFrame(numpy.power(10, probs), index=dataset.index, columns=[col + '_mixture'])
    data_table = pandas.concat([data_table, data_probs], axis=1)
    return data_table


def compute_chauvenet(dataset, columns: List[str], values: List[float]):
    for col in columns:
        for val in values:
            chauvenet_outliers = chauvenet(copy.deepcopy(dataset), col, val)
            visualizer.plot_binary_outliers(chauvenet_outliers, col, col + '_outlier')


def compute_mixture(dataset: pandas.DataFrame, columns: List[str], values: List[float]):
    for col in columns:
        for val in values:
            mixture_outliers = mixture_model(copy.deepcopy(dataset), col, val)
            visualizer.plot_dataset(mixture_outliers, [col, col + '_mixture'], ['exact','exact'], ['line', 'points'])


def compute_euclidean_distance(dataset: pandas.DataFrame, columns: List[str], values: List[list[float]]):
    # E.g. [[.5, .99], [.01, .99], [.1, .5]] where first value for each instance is less than the second value
    for col in columns:
        for val in values:
            dmin = val[0]
            fmin = val[1]
            dataset_outliers_sdb = outlier_techniques.simple_distance_based(copy.deepcopy(dataset), [col], 'euclidean', dmin, fmin)
            visualizer.plot_binary_outliers(dataset_outliers_sdb, col, 'simple_dist_outlier')


def compute_local_outlier(dataset: pandas.DataFrame, columns: List[str], values: List[float]):
    for col in columns:
        for val in values:
            dataset_outliers_lof = outlier_techniques.local_outlier_factor(copy.deepcopy(dataset), [col], 'euclidean', val)
            visualizer.plot_dataset(dataset_outliers_lof, [col, 'lof'], ['exact', 'exact'], ['line', 'points'])


if __name__ == '__main__':
    data = pandas.read_csv("../Python3Code/intermediate_datafiles/chapter2_result.csv", index_col=0)
    light_phone_col = "light_phone_lux"
    acc_phone_col = "acc_phone_x"

    print("Doing Chauvenet")
    compute_chauvenet(dataset=data, columns=[light_phone_col, acc_phone_col], values=[1,2,10])
    print("Doing Mixture Models")
    compute_mixture(dataset=data, columns=[light_phone_col, acc_phone_col], values=[1,3,10])
    print("Doing Euclidean Distance")
    compute_euclidean_distance(dataset=data, columns=[light_phone_col, acc_phone_col], values=[[0.01, 0.99],
                                                                                               [0.1, 0.99],
                                                                                               [0.5, 0.99]])
    print("Doing Local Outlier Factor")
    compute_local_outlier(dataset=data, columns=[light_phone_col, acc_phone_col], values=[2, 5, 10])
    print("DONE!")

