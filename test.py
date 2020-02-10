import numpy as np
from helpers import *
from classification import *
from eval import *
from cross_validation import *
from copy import *
from prune import *
import json

"""
    Common test codes
"""
# json intialisation
json_data = {}
json_data["root"] = []

# read datasets
train_attributes, train_labels = read_file("data/train_full.txt")
noisy_attributes, noisy_labels = read_file("data/train_noisy.txt")
sub_attributes, sub_labels = read_file("data/train_sub.txt")
test_attributes, test_labels = read_file("data/test.txt")
val_attributes, val_labels = hp.read_file("data/validation.txt")

# train
dtClassifier = DecisionTreeClassifier()
trained_tree = dtClassifier.train(train_attributes, train_labels)

# visualise
trained_tree.tree.print(0, json_data["root"])
with open('save.json', 'w') as outfile:
    json.dump(json_data, outfile)

# predict
predictions = dtClassifier.predict(test_attributes)

# evaluate
evaluator = Evaluator()
confusion = evaluator.confusion_matrix(predictions, test_labels)
print('\n')
print("Confusion matrix: ")
print(confusion)

print('\n')
print("Accuracy: ",
      evaluator.accuracy(confusion))
print("Recall: ",
    evaluator.recall(confusion))
print("Precision: ",
    evaluator.precision(confusion))
print("F1: ",
    evaluator.f1_score(confusion))

# cross validation plain
best_tree_plain, trees_plain = cross_validation("data/train_full.txt", 10)
print('\n')
print("Cross validation plain: ")
print("Best tree: ", best_tree_plain)
print("All trees: ", trees_plain)

# cross validation with hyperparameter tuning
best_tree_hp, trees_hp = cross_validation("data/train_full.txt", 10, True)
print('\n')
print("Cross validation with hyperparameter tuning: ")
print("Best tree: ", best_tree_hp)
print("All trees: ", trees_hp)

# cross validation with prune
best_tree_prune, trees_prune = cross_validation("data/train_full.txt", 10, False, "prune_more")
print('\n')
print("Cross validation with pruning: ")
print("Best tree: ", best_tree_prune)
print("All trees: ", trees_prune)

# cross validation with both hyperparameter and pruning
best_tree_all, trees_all = cross_validation("data/train_full.txt", 10, True, "prune")
print('\n')
print("Cross validation with pruning: ")
print("Best tree: ", best_tree_all)
print("All trees: ", trees_all)

# prune
print('\n')
print("Depth of unpruned tree: ", trained_tree.tree.get_depth(0))
prune(trained_tree, trained_tree.tree, val_attributes, val_labels)
print("Depth of pruned tree: ", trained_tree.tree.get_depth(0))


"""
    Test unit functions
"""
"""

def cross_validation_accuracy(test_attributes, evaluator):
    # predict
    predictions_cv, k_trees_cv = cross_validation("data/train_full.txt", 10)

    predictions_cv = predictions_cv.predict(test_attributes)

    predictions_k_trees_cv = []

    for tree in k_trees_cv:
        predictions_k_trees_cv.append(tree.predict(test_attributes))

    predictions_majority_k_trees_cv = vote_majority_label(
        predictions_k_trees_cv)

    # evaluate
    confusion_cv = evaluator.confusion_matrix(predictions_cv, test_labels)
    confusion_majority_k_trees_cv = evaluator.confusion_matrix(
        predictions_majority_k_trees_cv, test_labels)
    print("Accuracy when training with cross-validation: ",
          evaluator.accuracy(confusion_cv))
    print(
        "Accuracy when training with cross-validation merged predictions for k trees: ",
        evaluator.accuracy(confusion_majority_k_trees_cv))


def max_depth_test(dt_classifier, val_attributes, val_labels):
    # print out the depth of the given tree and the depth after pruning
    dt_classifier_copy = deepcopy(dt_classifier)
    print("Before pruning the maximum depth of tree is: ",
          dt_classifier_copy.tree.get_depth(0))
    prune_more(dt_classifier_copy, dt_classifier_copy.tree, val_attributes,
               val_labels)
    print("After pruning the maximum depth of tree is: ",
          dt_classifier_copy.tree.get_depth(0))


def accuracy_with_prune_more(dt_classifier, evaluator, val_attributes,
                             val_labels, test_attributes, test_labels):
    dt_classifier_copy = deepcopy(dt_classifier)
    prune_more(dt_classifier_copy, dt_classifier_copy.tree, val_attributes,
               val_labels)
    predictions = dt_classifier_copy.predict(test_attributes)
    confusion = evaluator.confusion_matrix(predictions, test_labels)
    print("Accuracy on test dataset after prune_more is: ",
          evaluator.accuracy(confusion))
    print("Recall on test dataset after prune_more is: ",
          evaluator.recall(confusion))
    print("Precision on test dataset after prune_more is: ",
          evaluator.precision(confusion))
    print("F1 on test dataset after prune_more is: ",
          evaluator.f1_score(confusion))


def accuracy_with_prune(dt_classifier, evaluator, val_attributes, val_labels,
                        test_attributes, test_labels):
    dt_classifier_copy = deepcopy(dt_classifier)
    prune(dt_classifier_copy, dt_classifier_copy.tree, val_attributes,
          val_labels)
    predictions = dt_classifier_copy.predict(test_attributes)
    confusion = evaluator.confusion_matrix(predictions, test_labels)
    print("Accuracy on test dataset after prune is: ",
          evaluator.accuracy(confusion))
    print("Recall on test dataset after prune is: ",
          evaluator.recall(confusion))
    print("Precision on test dataset after prune is: ",
          evaluator.precision(confusion))
    print("F1 on test dataset after prune is: ", evaluator.f1_score(confusion))


def accuracy_pre_prune(dt_classifier, evaluator, val_attributes, val_labels,
                       test_attributes, test_labels):
    dt_classifier_copy = deepcopy(dt_classifier)
    predictions = dt_classifier_copy.predict(test_attributes)
    confusion = evaluator.confusion_matrix(predictions, test_labels)
    print("Accuracy on test dataset is: ", evaluator.accuracy(confusion))
    print("Recall on test dataset is: ", evaluator.recall(confusion))
    print("Precision on test dataset is: ", evaluator.precision(confusion))
    print("F1 on test dataset is: ", evaluator.f1_score(confusion))


def accuracy_vs_max_depth_test(dt_classifier, evaluator, test_attributes,
                               test_labels):
    for i in range(20):
        dt_classifier_copy = deepcopy(dt_classifier)
        dt_classifier_copy = prune_to_max_depth(dt_classifier_copy, 20 - i)
        predictions = dt_classifier_copy.predict(test_attributes)
        confusion = evaluator.confusion_matrix(predictions, test_labels)
        print("Accuracy after pruning to depth:", 20 - i, " is ",
              evaluator.accuracy(confusion))


def parameter_tuning(dt_classifier, train_attributes, train_labels, evaluator,
                     val_attributes, val_labels):
    accuracy_list = []
    for max_share in np.arange(0.0, 0.01, 0.001):
        dt_classifier.set_max_share_hyperparameter(max_share)
        dt_classifier.train(train_attributes, train_labels)
        predictions = dt_classifier.predict(val_attributes)
        confusion = evaluator.confusion_matrix(predictions, val_labels)
        accuracy = evaluator.accuracy(confusion)
        accuracy_list.append([max_share, accuracy, dt_classifier.tree])

    # Finding the best accuracy
    best_max_share = accuracy_list[0][0]
    best_accuracy = accuracy_list[0][1]
    best_tree = accuracy_list[0][2]
    for j in range(1, len(accuracy_list)):
        if (accuracy_list[j][1] > best_accuracy):
            best_accuracy = accuracy_list[j][1]
            best_max_share = accuracy_list[j][0]
            best_tree = accuracy_list[j][2]
    print(accuracy_list)
    print("Best max_share hyperparameter is: ", best_max_share)
    print("Best accuracy is: ", best_accuracy)


#################################################################

# print("======= Test 1: cross validation with train_full.txt and print out the macro prediction accuracy =======")
# print()
# cross_validation_accuracy(test_attributes, evaluator)



# print("======= Test 2: parameter tuning =======")
# parameter_tuning(dtClassifier, train_attributes, train_labels, evaluator,
#                  val_attributes, val_labels)
# parameter_tuning(dtClassifier, noisy_attributes, noisy_labels, evaluator,
#                  val_attributes, val_labels)
# parameter_tuning(dtClassifier, sub_attributes, sub_labels, evaluator,
#                  val_attributes, val_labels)



# print("======= Test 3: print out evaluation metrics of different pruning techniques =======")
# accuracy_pre_prune(dtClassifier, evaluator, val_attributes, val_labels,
#                    test_attributes, test_labels)
# print("====================================")
# accuracy_with_prune(dtClassifier, evaluator, val_attributes, val_labels,
#                     test_attributes, test_labels)
# print("====================================")
# accuracy_with_prune_more(dtClassifier, evaluator, val_attributes, val_labels,
#                          test_attributes, test_labels)

# print(
#     "Test: compare the maximum depth of decision tree before and after pruning"
# )
# max_depth_test(dtClassifier, val_attributes, val_labels)

# print(
#     "Test: print out the predition accuracy when tree in different maximum depth"
# )
# accuracy_vs_max_depth_test(dtClassifier, evaluator, test_attributes,
#                            test_labels)



# print("======= Test 4: funcitonality of cross validation with or without pre-pruning and post pruning on =======")
# print("Cross validation with no pre-pruning and no post-pruning...")
# cross_validation("data/train_full.txt", 10, False, None)
# print("Cross validation with no pre-pruning and prune...")
# cross_validation("data/train_full.txt", 10, False, "prune")
# print("Cross validation with pre-pruning and no post-pruning...")
# cross_validation("data/train_full.txt", 10, True, None)
# print("Cross validation with pre-pruning and prune_more")
# cross_validation("data/train_full.txt", 10, True, "prune_more")
"""