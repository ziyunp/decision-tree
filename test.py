
import numpy as np
from helpers import *
from classification import *
from eval import *
from cross_validation import *
from copy import *
from prune import *
# import json

### Hardcoded data ###
# jsonData = {}
# jsonData["root"] = []
######################

# read datasets
train_attributes, train_labels = read_file("data/train_full.txt")
test_attributes, test_labels = read_file("data/test.txt")
val_attributes, val_labels = hp.read_file("data/validation.txt")

# train
dtClassifier = DecisionTreeClassifier()
dtClassifier.train(train_attributes, train_labels)

# predict
predictions = dtClassifier.predict(test_attributes)

# evaluate
evaluator = Evaluator()
confusion = evaluator.confusion_matrix(predictions, test_labels)
print("Accuracy when training on full dataset: ", evaluator.accuracy(confusion))

# with open('save.json', 'w') as outfile:
#     json.dump(json_data, outfile)

def max_depth_test(dt_classifier, val_attributes, val_labels):
    """
    print out the depth of the given tree and the depth after pruning
    """
    dt_classifier_copy = copy(dt_classifier)
    print("Before pruning the maximum depth of tree is: ", dt_classifier_copy.tree.get_depth(0))
    prune(dt_classifier_copy, dt_classifier_copy.tree, val_attributes, val_labels)
    print("After pruning the maximum depth of tree is: ", dt_classifier_copy.tree.get_depth(0))

def cross_validation_accuracy(test_attributes, evaluator):
    predictions_cross_validation = cross_validation("data/train_full.txt", 10).predict(test_attributes)
    confusion_cross_validation = evaluator.confusion_matrix(predictions_cross_validation, test_labels)
    print("Accuracy when training with cross-validation: ", evaluator.accuracy(confusion_cross_validation))

# cross_validation("data/train_full.txt", 10, True)

#################################################################

# test1
# max_depth_test(dtClassifier, val_attributes, val_labels)

# test2
# cross_validation_accuracy(test_attributes, evaluator)