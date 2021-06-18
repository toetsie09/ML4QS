import json
import pprint

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Chapter7.Evaluation import ClassificationEvaluation
from util.VisualizeDataset import VisualizeDataset


def metrics(labels_flat, pred_flat) -> dict:
    """Function to various metrics of our predictions vs labels"""
    print(json.dumps(classification_report(labels_flat, pred_flat, output_dict=True)))
    print("\n**** Classification report")
    print(classification_report(labels_flat, pred_flat))

    print("\n***Confusion matrix")
    pprint.pprint(confusion_matrix(pred_flat, labels_flat))

    plot_class_report = pd.DataFrame(classification_report(pred_flat, labels_flat))
    fig = plot_class_report.plot(kind='bar', x="dataframe_1", y="dataframe_2")  # bar can be replaced by
    fig.savefig("classif_report.png", dpi=200, format='png', bbox_inches='tight')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion_matrix(pred_flat, labels_flat))
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + unique_labels)
    ax.set_yticklabels([''] + unique_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("conf_matrix.png", dpi=200, format='png', bbox_inches='tight')

    info = {
        "classification_report": json.dumps(classification_report(labels_flat, pred_flat)),
        # "confusion_matrix": json.dumps(confusion_matrix(pred_flat, labels_flat))
    }
    return info


DataViz = VisualizeDataset(__file__)
prepare = PrepareDatasetForLearning()
learner = ClassificationAlgorithms()
eval = ClassificationEvaluation()
epochs = 5

dataset = pd.read_csv("final_v2.csv", index_col=False)
# dataset = dataset.drop(["time", "Unnamed: 0"], axis=1)
dataset.index = pd.to_datetime(dataset.index)
dataset = dataset.dropna()
labels = dataset["label"]
del dataset["label"]
unique_labels = labels.unique()
print(len(dataset.index))
# print(data)
# print(len(labels.index))


end_training_set = int(0.7 * len(dataset.index))
train_X = dataset[0: end_training_set]
train_y = labels[0: end_training_set]
test_X = dataset[end_training_set:len(dataset.index)]
test_y = labels[end_training_set:len(labels.index)]

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


for repeat in range(0, epochs):
    # Non-deterministic models
    print("Training NeuralNetwork run {} / {} ... ".format(repeat, epochs))
    class_train_y, class_test_y, _, _ = learner.feedforward_neural_network(
        train_X, train_y, test_X, gridsearch=True
    )
    performance_feed_train += eval.accuracy(train_y, class_train_y)
    performance_test = eval.accuracy(test_y, class_test_y)
    performance_feed_test += performance_test
    if not best_run_nn or best_run_nn[0] < performance_test:
        best_run_nn = [performance_test, [test_y, class_test_y]]


    print("Training RandomForest run {} / {} ... ".format(repeat, epochs))
    class_train_y, class_test_y, _, _ = learner.random_forest(
        train_X, train_y, test_X, gridsearch=True
    )
    performance_rand_for_train += eval.accuracy(train_y, class_train_y)
    performance_test = eval.accuracy(test_y, class_test_y)
    performance_rand_for_test += performance_test
    if not best_run_rf or best_run_rf[0] < performance_test:
        best_run_rf = [performance_test, [test_y, class_test_y]]


    print("Training SVM run {} / {}".format(repeat, epochs))
    class_train_y, class_test_y, _, _ = learner.support_vector_machine_with_kernel(
        train_X, train_y, test_X, gridsearch=True
    )
    performance_supp_vec_train += eval.accuracy(train_y, class_train_y)
    performance_test = eval.accuracy(test_y, class_test_y)
    performance_supp_vec_test += performance_test
    if not best_run_svm or best_run_svm[0] < performance_test:
        best_run_svm = [performance_test, [test_y, class_test_y]]


    # Deterministic models
    print("Training K-Nearest Neighbor run {} / {} ... ".format(repeat, epochs))
    class_train_y, class_test_y, _, _ = learner.k_nearest_neighbor(
        train_X, train_y, test_X, gridsearch=True)
    performance_train_knn += eval.accuracy(train_y, class_train_y)
    performance_test = eval.accuracy(test_y, class_test_y)
    performance_test_knn += performance_test
    if not best_run_knn or best_run_knn[0] < performance_test:
        best_run_knn = [performance_test, [test_y, class_test_y]]


    print("Training Descision Tree run {} / {} ... ".format(repeat, epochs))
    class_train_y, class_test_y, _, _ = learner.decision_tree(
        train_X, train_y, test_X, gridsearch=True
    )
    performance_tr_dt += eval.accuracy(train_y, class_train_y)
    performance_test = eval.accuracy(test_y, class_test_y)
    performance_te_dt += performance_test
    if not best_run_dt or best_run_dt[0] < performance_test:
        best_run_dt = [performance_test, [test_y, class_test_y]]


    print("Training Naive Bayes run {} / {} ... ".format(repeat, epochs))
    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.naive_bayes(
        train_X, train_y, test_X)
    performance_tr_nb += eval.accuracy(train_y, class_train_y)
    performance_test = eval.accuracy(test_y, class_test_y)
    performance_te_nb += performance_test
    if not best_run_dt or best_run_dt[0] < performance_test:
        best_run_dt = [performance_test, [test_y, class_test_y]]


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


print(f"Performance Feed Forward")
print(f"Train: {overall_performance_tr_nn}")
print(f"Test: {overall_performance_te_nn}")
print("Metrics")
metrics(best_run_nn[0][0], best_run_nn[0][1])
print()

print(f"Performance Random Forest")
print(f"Train: {overall_performance_tr_rf}")
print(f"Test: {overall_performance_te_rf}")
print("Metrics")
metrics(best_run_rf[0][0], best_run_rf[0][1])
print()

print(f"Performance Support Vector Machine")
print(f"Train: {overall_performance_tr_svm}")
print(f"Test: {overall_performance_te_svm}")
print("Metrics")
metrics(best_run_svm[0][0], best_run_svm[0][1])
print()

print(f"Performance K-Nearest Neighbor")
print(f"Train: {overall_performance_tr_knn}")
print(f"Test: {overall_performance_tr_knn}")
print("Metrics Feed Forward")
metrics(best_run_knn[0][0], best_run_knn[0][1])
print()

print(f"Performance Decision Tree")
print(f"Train: {overall_performance_tr_dt}")
print(f"Test: {overall_performance_tr_dt}")
print("Metrics Feed Forward")
metrics(best_run_dt[0][0], best_run_dt[0][1])
print()

print(f"Performance Naive Bayesan")
print(f"Train: {overall_performance_tr_nb}")
print(f"Test: {overall_performance_tr_nb}")
print("Metrics Feed Forward")
metrics(best_run_nb[0][0], best_run_nb[0][1])
print()
