import numpy as np
from Classification import *
from helpers import *
from eval import *

def split_dataset(filename, k):

    attributes, labels = read_file(filename)
    dataset = get_data(attributes, labels)
    length = len(dataset)

    if ((k <= 1) or (k > length)):
        print("Error: invalid k value")
        return

    subsets = []
    divider = length // k
    for i in range (k):
        subsets.append(dataset[i*divider:(i+1)*divider])

    return np.array(subsets)


def cross_validation(filename, k):

    subsets = split_dataset(filename, k)
    accuracy = [] 
    total = 0

    for i in range (k):
        training_dataset = []
        validation_dataset = []
        test_dataset = []
        test_dataset = subsets[i]
        for training_subset in np.delete(subsets, i, 0):
            if (len(validation_dataset) == 0):
                validation_dataset = training_subset
            elif (len(training_dataset) == 0):
                training_dataset = training_subset
            else:
                training_dataset = np.append(training_dataset, training_subset, axis=0)
        accuracy.append(run(training_dataset, validation_dataset, test_dataset))

    for i in range (len(accuracy)):
        total += 1 - accuracy[i]

    global_error = total / len(accuracy)

    return global_error

def run(training_dataset, validation_dataset, test_dataset):

    x,y = training_dataset[:,:-1], [chr(training_dataset[i][-1]) for i in range(len(training_dataset))]
    x_validation, y_validation = validation_dataset[:,:-1], [chr(validation_dataset[i][-1]) for i in range(len(validation_dataset))]
    x_test, y_test = test_dataset[:,:-1], [chr(test_dataset[i][-1]) for i in range(len(test_dataset))]

    # print("Training the decision tree...")
    classifier = DecisionTreeClassifier()
    classifier = classifier.train(x, y)

    # print("Testing the decision tree...")
    predictions = classifier.predict(x_test)
    # print("Predictions: {}".format(predictions))
    # print("y_test: ", y_test)

    # print("Evaluating test predictions...")
    evaluator = Evaluator()
    
    # requires str in both params
    confusion = evaluator.confusion_matrix(predictions, y_test)
    
    # print("Confusion matrix:")
    # print(confusion)

    accuracy = evaluator.accuracy(confusion)
    # print()
    # print("Accuracy: {}".format(accuracy))

    (p, macro_p) = evaluator.precision(confusion)
    (r, macro_r) = evaluator.recall(confusion)
    (f, macro_f) = evaluator.f1_score(confusion)

    # print()
    # print("Class: Precision, Recall, F1")
    # for (i, (p1, r1, f1)) in enumerate(zip(p, r, f)):
    #     print("{}: {:.2f}, {:.2f}, {:.2f}".format(class_labels[i], p1, r1, f1));
   
    # print() 
    # print("Macro-averaged Precision: {:.2f}".format(macro_p))
    # print("Macro-averaged Recall: {:.2f}".format(macro_r))
    # print("Macro-averaged F1")

    # print("======================")

    return accuracy

# print(splitDataset("data/toy.txt", 11))
# print("Error for k=", 2, "is", cross_validation("data/toy.txt", 2))
for i in range (2, 11):
    print("Error for k=", i, "is", cross_validation("data/simple1.txt", i))
    