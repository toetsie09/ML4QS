import json
import pprint

import pandas as pd
import seaborn
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from Assignment3.nn_model import network_training_testing
from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Chapter7.Evaluation import ClassificationEvaluation
from util.VisualizeDataset import VisualizeDataset


def metrics(labels_flat, pred_flat, name: str) -> dict:
    """Function to various metrics of our predictions vs labels"""
    class_report = classification_report(pred_flat, labels_flat, output_dict=True)
    print(json.dumps(class_report))
    print("\n**** Classification report")
    print(classification_report(labels_flat, pred_flat))

    print("\n***Confusion matrix")
    pprint.pprint(confusion_matrix(pred_flat, labels_flat))

    # plot_class_report = pd.DataFrame.from_dict()
    # print(plot_class_report)
    # print(plot_class_report.columns)
    # # fig = plot_class_report.plot(kind='bar', x="dataframe_1", y="dataframe_2")  # bar can be replaced by
    # fig = plot_class_report.plot.bar(rot=1)
    # fig.figure.savefig(name + "_classif_report.png", dpi=200, format='png', bbox_inches='tight')
    # plt.close()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    conf_matrix = pd.DataFrame(confusion_matrix(pred_flat, labels_flat), index=[0,1,2,3], columns=[0,1,2,3])
    plt.figure(figsize=(10,7))
    seaborn.heatmap(conf_matrix, annot=True)
    # cax = ax.matshow(conf_matrix, annot=True)
    # plt.title('Confusion matrix of the classifier')
    # fig.colorbar(cax)
    # ax.set_xticklabels(unique_labels)
    # ax.set_yticklabels(unique_labels)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    plt.savefig(name + "_conf_matrix.png", dpi=200, format='png', bbox_inches='tight')
    plt.close()
    info = {
        "classification_report": classification_report(labels_flat, pred_flat, output_dict=True),
        # "confusion_matrix": json.dumps(confusion_matrix(pred_flat, labels_flat))
    }
    return class_report["weighted avg"]["precision"]


def change_labels(data:pd.DataFrame):
    data["label"] = 0
    data.loc[data["labelCycling"] == 1, "label"] = 0
    data.loc[data["labelJumping-jacks"] == 1, "label"] = 1
    data.loc[data["labelSit-ups"] == 1, "label"] = 2
    data.loc[data["labelSquats"] == 1, "label"] = 3
    data = data.drop(["labelCycling", "labelJumping-jacks", "labelSit-ups", "labelSquats"], axis=1)
    # data = data.drop(['acc_x_temp_mean_ws_20',
    #    'acc_y_temp_mean_ws_20', 'acc_z_temp_mean_ws_20',
    #    'gyro_x_temp_mean_ws_20', 'gyro_y_temp_mean_ws_20',
    #    'gyro_z_temp_mean_ws_20', 'magnet_x_temp_mean_ws_20',
    #    'magnet_y_temp_mean_ws_20', 'magnet_z_temp_mean_ws_20'], axis=1)
    return data


DataViz = VisualizeDataset(__file__)
prepare = PrepareDatasetForLearning()
learner = ClassificationAlgorithms()
eval = ClassificationEvaluation()
epochs = 1

dataset = pd.read_csv("intermediate_datafiles/finaldatasets/dataset_final_everthing_nofreq.csv", index_col=False)
# print(dataset.columns)
dataset = change_labels(dataset)

dataset.index = pd.to_datetime(dataset.index)
dataset = dataset.dropna()
dataset = dataset.sample(frac=1).reset_index(drop=True)

labels = dataset["label"]
dataset = dataset.drop(["label", "Unnamed: 0"], axis=1)
unique_labels = labels.unique()

print(dataset.head(10))
print(len(dataset.index))
print(dataset.columns)
print(unique_labels)


end_training_set = int(0.6 * len(dataset.index))
train_X = dataset[0: end_training_set]
train_y = labels[0: end_training_set]
test_X = dataset[end_training_set:len(dataset.index)]
test_y = labels[end_training_set:len(labels.index)]

print(len(train_X.index))
print(len(test_X.index))

performance_feed_train = 0
performance_feed_test = 0

performance_rand_for_train = 0
performance_rand_for_test = 0

performance_supp_vec_train = 0
performance_supp_vec_test = 0

performance_train_knn = 0
performance_test_knn = 0

performance_tr_dt = 0
performance_te_dt = 0

performance_tr_nb = 0
performance_te_nb = 0


best_run_nn = []
best_run_rf = []
best_run_svm = []
best_run_knn = []
best_run_dt = []
best_run_nb = []

# Counterproof MLP manual model
network_training_testing(
    input_data=dataset,
    output_data=labels,
    input_size=len(dataset.columns),
    output_size=len(labels.unique()),
    epochs=20,
    learning_rate=0.0001,
    batch_size=1,
    # device = "cuda" if torch.cuda.is_available() is True else "cpu"
    device = "cpu"
)
# input()


for repeat in range(0, epochs):
    # Non-deterministic models
    print("\nTraining NeuralNetwork run {} / {} ... ".format(repeat, epochs))
    class_train_y, class_test_y, _, _ = learner.feedforward_neural_network(
        train_X, train_y, test_X, gridsearch=True
    )
    performance_feed_train += eval.accuracy(train_y, class_train_y)
    performance_test = eval.accuracy(test_y, class_test_y)
    performance_feed_test += performance_test
    if not best_run_nn or best_run_nn[0] < performance_test:
        best_run_nn = [performance_test, [test_y, class_test_y]]
    print(f"Performance: {performance_test}")


    print("\nTraining RandomForest run {} / {} ... ".format(repeat, epochs))
    class_train_y, class_test_y, _, _ = learner.random_forest(
        train_X, train_y, test_X, gridsearch=True
    )
    performance_rand_for_train += eval.accuracy(train_y, class_train_y)
    performance_test = eval.accuracy(test_y, class_test_y)
    performance_rand_for_test += performance_test
    if not best_run_rf or best_run_rf[0] < performance_test:
        best_run_rf = [performance_test, [test_y, class_test_y]]
    print(f"Performance: {performance_test}")


    print("\nTraining SVM run {} / {}".format(repeat, epochs))
    class_train_y, class_test_y, _, _ = learner.support_vector_machine_with_kernel(
        train_X, train_y, test_X, gridsearch=True
    )
    performance_supp_vec_train += eval.accuracy(train_y, class_train_y)
    performance_test = eval.accuracy(test_y, class_test_y)
    performance_supp_vec_test += performance_test
    if not best_run_svm or best_run_svm[0] < performance_test:
        best_run_svm = [performance_test, [test_y, class_test_y]]
    print(f"Performance: {performance_test}")


    # Deterministic models
    print("\nTraining K-Nearest Neighbor run {} / {} ... ".format(repeat, epochs))
    class_train_y, class_test_y, _, _ = learner.k_nearest_neighbor(
        train_X, train_y, test_X, gridsearch=True)
    performance_train_knn += eval.accuracy(train_y, class_train_y)
    performance_test = eval.accuracy(test_y, class_test_y)
    performance_test_knn += performance_test
    if not best_run_knn or best_run_knn[0] < performance_test:
        best_run_knn = [performance_test, [test_y, class_test_y]]
    print(f"Performance: {performance_test}")


    print("\nTraining Descision Tree run {} / {} ... ".format(repeat, epochs))
    class_train_y, class_test_y, _, _ = learner.decision_tree(
        train_X=train_X, train_y=train_y, test_X=test_X, gridsearch=True, export_tree_path=".")
    performance_tr_dt += eval.accuracy(train_y, class_train_y)
    performance_test = eval.accuracy(test_y, class_test_y)
    performance_te_dt += performance_test
    if not best_run_dt or best_run_dt[0] < performance_test:
        best_run_dt = [performance_test, [test_y, class_test_y]]
    print(f"Performance: {performance_test}")


    print("\nTraining Naive Bayes run {} / {} ... ".format(repeat, epochs))
    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.naive_bayes(
        train_X, train_y, test_X)
    performance_tr_nb += eval.accuracy(train_y, class_train_y)
    performance_test = eval.accuracy(test_y, class_test_y)
    performance_te_nb += performance_test
    if not best_run_nb or best_run_nb[0] < performance_test:
        best_run_nb = [performance_test, [test_y, class_test_y]]
    print(f"Performance: {performance_test}")

overall_performance_tr_nn = performance_feed_train / epochs
overall_performance_te_nn = performance_feed_test / epochs

overall_performance_tr_rf = performance_rand_for_train / epochs
overall_performance_te_rf = performance_rand_for_test / epochs

overall_performance_tr_svm = performance_supp_vec_train / epochs
overall_performance_te_svm = performance_supp_vec_test / epochs

overall_performance_tr_knn = performance_train_knn / epochs
overall_performance_te_knn = performance_test_knn / epochs

overall_performance_tr_dt = performance_tr_dt / epochs
overall_performance_te_dt = performance_te_dt / epochs

overall_performance_tr_nb = performance_tr_nb / epochs
overall_performance_te_nb = performance_tr_nb / epochs

weighted_prec_averages = []

print(f"Performance Feed Forward")
print(f"Train: {overall_performance_tr_nn}")
print(f"Test: {overall_performance_te_nn}")
print("Metrics")
weighted_prec_averages.append(metrics(best_run_nn[1][0], best_run_nn[1][1], "FeedForward"))
print()

print(f"Performance Random Forest")
print(f"Train: {overall_performance_tr_rf}")
print(f"Test: {overall_performance_te_rf}")
print("Metrics RF")
weighted_prec_averages.append(metrics(best_run_rf[1][0], best_run_rf[1][1], "RandomForest"))
print()

print(f"Performance Support Vector Machine")
print(f"Train: {overall_performance_tr_svm}")
print(f"Test: {overall_performance_te_svm}")
print("Metrics SVM")
weighted_prec_averages.append(metrics(best_run_svm[1][0], best_run_svm[1][1], "SVM"))
print()

print(f"Performance K-Nearest Neighbor")
print(f"Train: {overall_performance_tr_knn}")
print(f"Test: {overall_performance_tr_knn}")
print("Metrics KNN")
weighted_prec_averages.append(metrics(best_run_knn[1][0], best_run_knn[1][1], "KNN"))
print()

print(f"Performance Decision Tree")
print(f"Train: {overall_performance_tr_dt}")
print(f"Test: {overall_performance_tr_dt}")
print("Metrics Decision Tree")
weighted_prec_averages.append(metrics(best_run_dt[1][0], best_run_dt[1][1], "DT"))
print()

print(f"Performance Naive Bayesan")
print(f"Train: {overall_performance_tr_nb}")
print(f"Test: {overall_performance_tr_nb}")
print("Metrics Naive Bayes")
weighted_prec_averages.append(metrics(best_run_nb[1][0], best_run_nb[1][1], "NB"))
print()


print(weighted_prec_averages)
plt.close()
# plt.bar(["SupportVM", "K-NN", "DecisionT", "NaiveB"], weighted_prec_averages)
plt.bar(["FeedF","RandomF", "SupportVM", "K-NN", "DecisionT", "NaiveB"], weighted_prec_averages)
plt.show()