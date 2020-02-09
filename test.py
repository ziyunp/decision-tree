import numpy as np
from helpers import *
from classification import *
from eval import *
from cross_validation import *
from copy import *
from prune import *
# import json

"""
    Common test codes
"""
# json intialisation
json_data = {}
json_data["root"] = []

# read datasets
train_attributes, train_labels = read_file("data/train_full.txt")
noisy_attributes, noisy_labels = read_file("data/train_noisy.txt")
test_attributes, test_labels = read_file("data/test.txt")
val_attributes, val_labels = hp.read_file("data/validation.txt")

# train
dtClassifier = DecisionTreeClassifier()
# dtClassifier.set_max_share_hyperparameter(0.004)
dtClassifier.train(train_attributes, train_labels)

# predict
predictions = dtClassifier.predict(test_attributes)

# evaluate
evaluator = Evaluator()
confusion = evaluator.confusion_matrix(predictions, test_labels)
print("Accuracy when training on full dataset: ", evaluator.accuracy(confusion))

with open('save.json', 'w') as outfile:
    json.dump(json_data, outfile)



"""
    Test unit functions
"""
def cross_validation_accuracy(test_attributes, evaluator):
    # prefict
    predictions_cross_validation, k_trees_cross_validation = cross_validation("data/train_full.txt", 10)

    predictions_cross_validation = predictions_cross_validation.predict(test_attributes)

    predictions_k_trees_cross_validation = []

    for tree in k_trees_cross_validation:
        predictions_k_trees_cross_validation.append(tree.predict(test_attributes))

    predictions_majority_k_trees_cross_validation = get_majority_label_cross_validation(predictions_k_trees_cross_validation)

    # evaluate
    confusion_cross_validation = evaluator.confusion_matrix(predictions_cross_validation, test_labels)
    confusion_majority_k_trees_cross_validation = evaluator.confusion_matrix(predictions_majority_k_trees_cross_validation, test_labels)
    print("Accuracy when training with cross-validation: ", evaluator.accuracy(confusion_cross_validation))
    print("Accuracy when training with cross-validation merged predictions for k trees: ", evaluator.accuracy(confusion_majority_k_trees_cross_validation))


def max_depth_test(dt_classifier, val_attributes, val_labels):
    """
    print out the depth of the given tree and the depth after pruning
    """
    dt_classifier_copy = deepcopy(dt_classifier)
    print("Before pruning the maximum depth of tree is: ", dt_classifier_copy.tree.get_depth(0))
    prune_more(dt_classifier_copy, dt_classifier_copy.tree, val_attributes, val_labels)
    print("After pruning the maximum depth of tree is: ", dt_classifier_copy.tree.get_depth(0))


def cross_validation_accuracy_after_pruning_more():
    cross_validation("data/train_full.txt", 10, False, "prune_more")


def cross_validation_accuracy_after_pruning():
    cross_validation("data/train_full.txt", 10, False, "prune")


# def cross_validation_accuracy():
#     cross_validation("data/train_full.txt", 10, False, None)


def accuracy_with_prune_more(dt_classifier, evaluator, val_attributes, val_labels, test_attributes, test_labels):
    dt_classifier_copy = deepcopy(dt_classifier)
    prune_more(dt_classifier_copy, dt_classifier_copy.tree, val_attributes, val_labels)
    predictions = dt_classifier_copy.predict(test_attributes)
    confusion = evaluator.confusion_matrix(predictions, test_labels)
    print("Accuracy on test dataset after prune_more is: ", evaluator.accuracy(confusion))
    print("Recall on test dataset after prune_more is: ", evaluator.recall(confusion))
    print("Precision on test dataset after prune_more is: ", evaluator.precision(confusion))


def accuracy_with_prune(dt_classifier, evaluator, val_attributes, val_labels, test_attributes, test_labels):
    dt_classifier_copy = deepcopy(dt_classifier)
    prune(dt_classifier_copy, dt_classifier_copy.tree, val_attributes, val_labels)
    predictions = dt_classifier_copy.predict(test_attributes)
    confusion = evaluator.confusion_matrix(predictions, test_labels)
    print("Accuracy on test dataset after prune is: ", evaluator.accuracy(confusion))
    print("Recall on test dataset after prune is: ", evaluator.recall(confusion))
    print("Precision on test dataset after prune is: ", evaluator.precision(confusion))


def accuracy_pre_prune(dt_classifier, evaluator, val_attributes, val_labels, test_attributes, test_labels):
    dt_classifier_copy = deepcopy(dt_classifier)
    predictions = dt_classifier_copy.predict(test_attributes)
    confusion = evaluator.confusion_matrix(predictions, test_labels)
    print("Accuracy on test dataset is: ", evaluator.accuracy(confusion))
    print("Recall on test dataset is: ", evaluator.recall(confusion))
    print("Precision on test dataset is: ", evaluator.precision(confusion))


def accuracy_vs_max_depth_test(dt_classifier, evaluator, test_attributes, test_labels):
    for i in range(20):
        dt_classifier_copy = deepcopy(dt_classifier)
        dt_classifier_copy = prune_to_max_depth(dt_classifier_copy, 20 - i)
        predictions = dt_classifier_copy.predict(test_attributes)
        confusion = evaluator.confusion_matrix(predictions, test_labels)
        print("Accuracy after pruning to depth:", 20 - i, " is ", evaluator.accuracy(confusion))


#################################################################

# test1
# max_depth_test(dtClassifier, val_attributes, val_labels)

# print(dtClassifier.tree.split_info.attribute)
# print(dtClassifier.tree.split_info.value)
# print(dtClassifier.tree.child_true.get_cur_freq())
# print(dtClassifier.tree.child_false.get_cur_freq())

# test2
# cross_validation_accuracy(test_attributes, evaluator)

# test3
# cross_validation_accuracy_after_pruning_more()
# cross_validation_accuracy_after_pruning()
# cross_validation_accuracy()

# test5
# accuracy_pre_prune(dtClassifier, evaluator, val_attributes, val_labels, test_attributes, test_labels)
# print("====================================")
# accuracy_with_prune(dtClassifier, evaluator, val_attributes, val_labels, test_attributes, test_labels)
# print("====================================")
# accuracy_with_prune_more(dtClassifier, evaluator, val_attributes, val_labels, test_attributes, test_labels)

# test6
# accuracy_vs_max_depth_test(dtClassifier, evaluator, test_attributes, test_labels)

# Test: cross validation with train_full.txt and print out the macro prediction accuracy
print()
print("======= Cross validation with NO post-pruning ========")
cross_validation_accuracy(test_attributes, evaluator)
