import pandas as pd
from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Chapter7.Evaluation import ClassificationEvaluation
from util.VisualizeDataset import VisualizeDataset
from sklearn.impute import SimpleImputer
import numpy

imp = SimpleImputer(strategy='most_frequent')
imp.fit_transform()
DataViz = VisualizeDataset(__file__)
prepare = PrepareDatasetForLearning()
learner = ClassificationAlgorithms()
eval = ClassificationEvaluation()
epochs = 5

dataset = pd.read_csv("final_v2.csv", index_col=False)
# dataset = dataset.drop(["time", "Unnamed: 0"], axis=1)
dataset.index = pd.to_datetime(dataset.index)
dataset = dataset.dropna()
print(len(dataset.index))
labels = dataset["label"]
data = dataset.drop(["label"], axis=1)
# print(data)
# print(len(labels.index))


end_training_set = int(0.7 * len(dataset.index))
train_X = data[0: end_training_set]
train_y = labels[0: end_training_set]
test_X = data[end_training_set:len(data.index)]
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


for repeat in range(0, epochs):
    # Non-deterministic models
    print("Training NeuralNetwork run {} / {} ... ".format(repeat, epochs))
    train_X = train_X.fillna(0)
    test_X = test_X.fillna(0)
    train_y = train_y.fillna(0)
    test_y = test_y.fillna(0)
    class_train_y, class_test_y, _, _ = learner.feedforward_neural_network(
        train_X, train_y, test_X, gridsearch=True
    )
    performance_feed_train += eval.accuracy(train_y, class_train_y)
    performance_feed_test += eval.accuracy(test_y, class_test_y)


    print("Training RandomForest run {} / {} ... ".format(repeat, epochs))
    train_X = train_X.fillna(0)
    test_X = test_X.fillna(0)
    train_y = train_y.fillna(0)
    test_y = test_y.fillna(0)
    class_train_y, class_test_y, _, _ = learner.random_forest(
        train_X, train_y, test_X, gridsearch=True
    )
    performance_rand_for_train += eval.accuracy(train_y, class_train_y)
    performance_rand_for_test += eval.accuracy(test_y, class_test_y)


    print("Training SVM run {} / {}".format(repeat, epochs))
    train_X = train_X.fillna(0)
    test_X = test_X.fillna(0)
    train_y = train_y.fillna(0)
    test_y = test_y.fillna(0)
    class_train_y, class_test_y, _, _ = learner.support_vector_machine_with_kernel(
        train_X, train_y, test_X, gridsearch=True
    )
    performance_supp_vec_train += eval.accuracy(train_y, class_train_y)
    performance_supp_vec_test += eval.accuracy(test_y, class_test_y)


    # Deterministic models
    print("Training Nearest Neighbor run {} / {} ... ".format(repeat, epochs))
    train_X = train_X.fillna(0)
    test_X = test_X.fillna(0)
    train_y = train_y.fillna(0)
    test_y = test_y.fillna(0)
    class_train_y, class_test_y, _, _ = learner.k_nearest_neighbor(
        train_X, train_y, test_X, gridsearch=True)
    performance_train_knn += eval.accuracy(train_y, class_train_y)
    performance_test_knn += eval.accuracy(test_y, class_test_y)


    print("Training Descision Tree run {} / {} ... ".format(repeat, epochs))
    train_X = train_X.fillna(0)
    test_X = test_X.fillna(0)
    train_y = train_y.fillna(0)
    test_y = test_y.fillna(0)
    class_train_y, class_test_y, _, _ = learner.decision_tree(
        train_X, train_y, test_X, gridsearch=True
    )
    performance_tr_dt += eval.accuracy(train_y, class_train_y)
    performance_te_dt += eval.accuracy(test_y, class_test_y)
    
    
    print("Training Naive Bayes run {} / {} ... ".format(repeat, epochs))
    train_X = train_X.fillna(0)
    test_X = test_X.fillna(0)
    train_y = train_y.fillna(0)
    test_y = test_y.fillna(0)
    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.naive_bayes(
        train_X, train_y, test_X)
    performance_tr_nb += eval.accuracy(train_y, class_train_y)
    performance_te_nb += eval.accuracy(test_y, class_test_y)


overall_performance_tr_nn = performance_feed_train/epochs
overall_performance_te_nn = performance_feed_test/epochs

overall_performance_tr_rf = performance_rand_for_train/epochs
overall_performance_te_rf = performance_rand_for_test/epochs

overall_performance_tr_svm = performance_supp_vec_train/epochs
overall_performance_te_svm = performance_supp_vec_test/epochs

overall_performance_tr_knn = performance_train_knn/epochs
overall_performance_te_knn = performance_test_knn/epochs

overall_performance_tr_dt = performance_tr_dt/epochs
overall_performance_te_dt = performance_te_dt/epochs

overall_performance_tr_nb = performance_tr_nb/epochs
overall_performance_te_nb = performance_tr_nb/epochs


print(f"performance feed forward")
print(f"Train: {overall_performance_tr_nn}")
print(f"Test: {overall_performance_te_nn}")

print(f"performance random foresr")
print(f"Train: {overall_performance_tr_rf}")
print(f"Test: {overall_performance_te_rf}")

print(f"performance support vector machine")
print(f"Train: {overall_performance_tr_svm}")
print(f"Test: {overall_performance_te_svm}")

print(f"performance k-nearest neighbor")
print(f"Train: {overall_performance_tr_knn}")
print(f"Test: {overall_performance_tr_knn}")

print(f"performance decision tree")
print(f"Train: {overall_performance_tr_dt}")
print(f"Test: {overall_performance_tr_dt}")

print(f"performance naive bayesan")
print(f"Train: {overall_performance_tr_nb}")
print(f"Test: {overall_performance_tr_nb}")
