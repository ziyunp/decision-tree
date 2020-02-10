import numpy as np
from helpers import *
from classification import *
from eval import *
from prune import *

HYPERPARAMETER_THRESHOLD = 0.01
STEP = 0.002


def split_dataset(dataset, k):
    subsets = []
    divider = len(dataset) // k
    for i in range(k):
        subsets.append(dataset[i * divider:(i + 1) * divider])

    return np.array(subsets)


# This function allows optional hyperparameter tuning and pruning
def cross_validation(filename,
                     k,
                     hyperparameter_tuning=False,
                     prune_func=None):

    attributes, labels = read_file(filename)
    dataset = get_data(attributes, labels)
    length = len(dataset)

    # check if k value is valid
    min_k = 2
    if hyperparameter_tuning:
        min_k = 3
    if ((k < min_k) or (k > length)):
        print("Error: invalid k value")
        return

    # shuffle dataset
    np.random.shuffle(dataset)
    # split dataset
    subsets = split_dataset(dataset, k)
    models_list = []
    all_trees = []

    for i in range(k - 1):
        accuracy_list = []
        training_dataset = []
        test_dataset = []

        # only need a validation dataset if hyperparameter_tuning or pruning
        if hyperparameter_tuning or not prune_func == None:
            validation_dataset = []

        # divide subsets into test, validation and training sets
        # ratio of test:validation:training is 1:1:k-2 if hyperparameter_tuning
        # otherwise ratio is 1:k-1
        test_dataset = subsets[i]
        for training_subset in np.delete(subsets, i, 0):
            if (hyperparameter_tuning or
                    not prune_func == None) and len(validation_dataset) == 0:
                validation_dataset = training_subset
            elif (len(training_dataset) == 0):
                training_dataset = training_subset
            else:
                training_dataset = np.append(training_dataset,
                                             training_subset,
                                             axis=0)

        if hyperparameter_tuning:
            # Try hyperparameters from 0 to HYPERPARAMETER_THRESHOLD
            # at an interval of STEP
            for max_share in np.arange(0.00, HYPERPARAMETER_THRESHOLD, STEP):
                accuracy, tree = run(training_dataset, validation_dataset,
                                     max_share, prune_func)
                accuracy_list.append([max_share, accuracy, tree])

            best_max_share = accuracy_list[0][0]
            best_accuracy = accuracy_list[0][1]
            best_tree = accuracy_list[0][2]

            # Find best hyperparameter by the accuracy of predictions on the validation set
            for j in range(1, len(accuracy_list)):
                if (accuracy_list[j][1] > best_accuracy):
                    best_accuracy = accuracy_list[j][1]
                    best_max_share = accuracy_list[j][0]
                    best_tree = accuracy_list[j][2]

            print("Best max_share hyperparameter is: ", best_max_share)
            print("Best accuracy is: ", best_accuracy)

            accuracy_test = get_test_accuracy(best_tree, test_dataset)

            print("Accuracy on the test dataset with the best max_share is: ",
                  accuracy_test)

            all_trees.append(best_tree)
            models_list.append([accuracy_test, best_tree])

        # no hyperparameter tuning but with pruning
        elif (not prune_func == None):
            accuracy_val, tree = run(training_dataset, validation_dataset, 0,
                                     prune_func)
            print("With pruning: ", prune_func,
                  "the accuracy on validation set is: ", accuracy_val)
            accuracy_test = get_test_accuracy(tree, test_dataset)

            all_trees.append(tree)
            models_list.append([accuracy_test, tree])

        # train without hyperparameter tuning nor pruning
        else:
            accuracy_test, tree = run(training_dataset, test_dataset, 0, None)
            all_trees.append(tree)
            models_list.append([accuracy_test, tree])

    best_accuracy = models_list[0][0]
    best_tree = models_list[0][1]
    # Find the best accuracy
    for j in range(1, len(models_list)):
        if (models_list[j][0] > best_accuracy):
            best_accuracy = models_list[j][0]
            best_tree = models_list[j][1]

    average_accuracy = np.average(
        [models_list[i][0] for i in range(len(models_list))])
    standard_deviation = np.std(
        [models_list[i][0] for i in range(len(models_list))])
    print("Standard deviation: ", average_accuracy, " +- ", standard_deviation)

    return best_tree, all_trees


def run(training_dataset, test_dataset, max_share, prune_func):
    x, y = training_dataset[:, :-1], [
        chr(training_dataset[i][-1]) for i in range(len(training_dataset))
    ]
    if not prune_func == None:
        x_test, y_test = test_dataset[:, :-1], [
            chr(test_dataset[i][-1]) for i in range(len(test_dataset))
        ]

    classifier = DecisionTreeClassifier()
    classifier.set_max_share_hyperparameter(max_share)
    classifier = classifier.train(x, y)

    if (prune_func == "prune"):
        prune(classifier, classifier.tree, x_test, y_test)
    elif (prune_func == "prune_more"):
        prune_more(classifier, classifier.tree, x_test, y_test)

    accuracy = get_test_accuracy(classifier, test_dataset)

    return accuracy, classifier


def get_test_accuracy(classifier, test_dataset):
    x_test, y_test = test_dataset[:, :-1], [
        chr(test_dataset[i][-1]) for i in range(len(test_dataset))
    ]
    predictions = classifier.predict(x_test)
    evaluator = Evaluator()
    confusion = evaluator.confusion_matrix(predictions, y_test)
    return evaluator.accuracy(confusion)
