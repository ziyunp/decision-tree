import numpy as np
from classification import *
from helpers import *
from eval import *

def splitDataset(filename, k):

    attributes, labels = readFile(filename)
    dataset = getData(attributes, labels)
    length = len(dataset)

    if ((k <= 0) or (k > length)):
        print("Error: invalid k value")
        return

    subsets = []
    divider = int(length / k)

    for i in range (k):
        if (i == (k - 1)):
            subsets.append(dataset[(k-1)*divider:])
        else :
            subsets.append(dataset[i*divider:(i+1)*divider])

    return np.array(subsets)


def crossValidation(filename, k):

    subsets = splitDataset(filename, k)
    accuracy = [] 
    total = 0

    for i in range (k):
        testDataset = []
        trainingDataset = []

        testDataset = subsets[i]

        for trainingSubset in np.delete(subsets, i, 0):
            if (len(trainingDataset) == 0):
                trainingDataset = trainingSubset
            else:
                trainingDataset = np.append(trainingDataset, trainingSubset, axis=0)
            
        accuracy.append(run(trainingDataset, testDataset))

    for i in range (len(accuracy)):
        total += 1 - accuracy[i]

    globalError = total / len(accuracy)

    return globalError

def run(trainingDataset, testDataset):

    x,y = trainingDataset[:,:-1], trainingDataset[:,-1]
    x_test, y_test = testDataset[:,:-1], testDataset[:,-1]

    # print("Training the decision tree...")
    classifier = DecisionTreeClassifier()
    classifier = classifier.train(x, y)

    # print("Testing the decision tree...")
    predictions = classifier.predict(x_test)
    # print("Predictions: {}".format(predictions))
    # print("y_test: ", y_test)

    # print("Evaluating test predictions...")
    evaluator = Evaluator()
    classLabels = classLabels = np.unique(predictions)
    confusion = evaluator.confusion_matrix(predictions, y_test, classLabels)
    
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
    #     print("{}: {:.2f}, {:.2f}, {:.2f}".format(classLabels[i], p1, r1, f1));
   
    # print() 
    # print("Macro-averaged Precision: {:.2f}".format(macro_p))
    # print("Macro-averaged Recall: {:.2f}".format(macro_r))
    # print("Macro-averaged F1")

    # print("======================")

    return accuracy

# print(splitDataset("data/toy.txt", 11))
for i in range (2, 11):
    print("Error for k=", i, "is", crossValidation("data/simple1.txt", i))
    