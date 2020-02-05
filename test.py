
import numpy as np
from helpers import *
from classification import *
from eval import *
from cross_validation import *
# import json

### Hardcoded data ###
# jsonData = {}
# jsonData["root"] = []
######################

train_attributes, train_labels = read_file("data/train_full.txt")
dtClassifier = DecisionTreeClassifier()
dtClassifier.train(train_attributes, train_labels)

test_attributes, test_labels = read_file("data/test.txt")
predictions = dtClassifier.predict(test_attributes)
predictions_cross_validation = cross_validation("data/train_full.txt", 10).predict(test_attributes)

evaluator = Evaluator()
confusion = evaluator.confusion_matrix(predictions, test_labels)
confusion_cross_validation = evaluator.confusion_matrix(predictions_cross_validation, test_labels)
print("Accuracy when training on full dataset: ", evaluator.accuracy(confusion))
print("Accuracy when training with cross-validation: ", evaluator.accuracy(confusion_cross_validation))

# with open('save.json', 'w') as outfile:
#     json.dump(json_data, outfile)
