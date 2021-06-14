
from Chapter7.Evaluation import RegressionEvaluation
from Chapter8.LearningAlgorithmsTemporal import TemporalClassificationAlgorithms
from Chapter8.LearningAlgorithmsTemporal import TemporalRegressionAlgorithms

import pandas as pd



dataset = pd.read_csv("final_v2.csv", index_col=False)
# dataset = dataset.drop(["time", "Unnamed: 0"], axis=1)
dataset.index = pd.to_datetime(dataset.index)
dataset = dataset.dropna()
# print(len(dataset.index))
labels = dataset["label"]
labels.loc[labels == "Walking",] = 0
labels.loc[labels == "Sitting",] = 1
labels.loc[labels == "Sitting",] = 2
labels.loc[labels == "Yardwork",] = 3
labels.loc[labels == "OnTable",] = 4
labels.loc[labels == "Cycling",] = 5
labels.loc[labels == "ScreenTime",] = 6
labels = labels[labels!="Name"]
# print(labels)
# print(labels.unique())
# quit()
data = dataset.drop(["label"], axis=1)


end_training_set = int(0.7 * len(dataset.index))
train_X = data[0: end_training_set]
train_y = labels[0: end_training_set]
test_X = data[end_training_set:len(data.index)]
test_y = labels[end_training_set:len(labels.index)]

learner = TemporalRegressionAlgorithms()
eval = RegressionEvaluation()

# We repeat the experiment a number of times to get a bit more robust data as the initialization of e.g. the NN is random.

repeats = 10

# we set a washout time to give the NN's the time to stabilize. We do not compute the error during the washout time.

washout_time = 10

scores_over_all_algs = []

performance_tr_res = 0
performance_tr_res_std = 0
performance_te_res = 0
performance_te_res_std = 0
performance_tr_rnn = 0
performance_tr_rnn_std = 0
performance_te_rnn = 0
performance_te_rnn_std = 0

for repeat in range(0, repeats):
    print(f'---- run {repeat} ---')
    regr_train_y, regr_test_y = learner.reservoir_computing(train_X, train_y, test_X, test_y, gridsearch=True, per_time_step=False)

    mean_tr, std_tr = eval.mean_squared_error_with_std(train_y.iloc[washout_time:,], regr_train_y.iloc[washout_time:,])
    mean_te, std_te = eval.mean_squared_error_with_std(test_y.iloc[washout_time:,], regr_test_y.iloc[washout_time:,])

    performance_tr_res += mean_tr
    performance_tr_res_std += std_tr
    performance_te_res += mean_te
    performance_te_res_std += std_te

    regr_train_y, regr_test_y = learner.recurrent_neural_network(train_X, train_y, test_X, test_y, gridsearch=True)

    mean_tr, std_tr = eval.mean_squared_error_with_std(train_y.iloc[washout_time:,], regr_train_y.iloc[washout_time:,])
    mean_te, std_te = eval.mean_squared_error_with_std(test_y.iloc[washout_time:,], regr_test_y.iloc[washout_time:,])

    performance_tr_rnn += mean_tr
    performance_tr_rnn_std += std_tr
    performance_te_rnn += mean_te
    performance_te_rnn_std += std_te

