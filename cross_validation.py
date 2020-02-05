import numpy as np
from helpers import *
from Classification import *
import Node as nd
from eval import *

def split_dataset(filename, k):

    attributes, labels = read_file(filename)
    dataset = get_data(attributes, labels)
    length = len(dataset)

    if ((k <= 1) or (k > length)):
        print("Error: invalid k value")
        return

    # Shuffle
    np.random.shuffle(dataset)
    subsets = []
    divider = length // k
    for i in range (k):
        subsets.append(dataset[i*divider:(i+1)*divider])

    return np.array(subsets)


def cross_validation(filename, k, hyperparameter_tuning = False):

    subsets = split_dataset(filename, k)
    models_list = [] 

    for i in range (k - 1):
        accuracy_list = [] 
        training_dataset = []
        if hyperparameter_tuning:
            validation_dataset = []
        test_dataset = []
        test_dataset = subsets[i]
        for training_subset in np.delete(subsets, i, 0):
            if (hyperparameter_tuning and len(validation_dataset) == 0):
                validation_dataset = training_subset
            elif (len(training_dataset) == 0):
                training_dataset = training_subset
            else:
                training_dataset = np.append(training_dataset, training_subset, axis=0)
        

        if hyperparameter_tuning:
            # Trying different hyperparameters
            for max_share in np.arange (0.00, 0.01, 0.002):
                accuracy, tree = run(training_dataset, validation_dataset, max_share)
                accuracy_list.append([max_share, accuracy, tree])

            best_max_share = accuracy_list[0][0]
            best_accuracy = accuracy_list[0][1]
            best_tree = accuracy_list[0][2]

            # Finding the best accuracy
            for j in range (1, len(accuracy_list)):
                if (accuracy_list[j][1] > best_accuracy):
                    best_accuracy = accuracy_list[j][1]
                    best_max_share = accuracy_list[j][0]
                    best_tree = accuracy_list[j][2]

            print("Best max_share hyperparameter is: ", best_max_share)
            print("Best accuracy is: ", best_accuracy)

            accuracy_test, tree = run(training_dataset, test_dataset, best_max_share)
            print("Accuracy on the test dataset with the best max_share is: ", accuracy_test)

            models_list.append([accuracy_test, tree])

            print("======================")


        else:
            accuracy_test, tree = run(training_dataset, test_dataset, 0)
            models_list.append([accuracy_test, tree])

    best_accuracy = models_list[0][0]
    best_tree = models_list[0][1]
    # Finding the best accuracy
    for j in range (1, len(models_list)):
        if (models_list[j][0] > best_accuracy):
            best_accuracy = models_list[j][0]
            best_tree = models_list[j][1]

    average_accuracy = np.average([models_list[i][0] for i in range (len(models_list))])
    standard_deviation = np.std([models_list[i][0] for i in range (len(models_list))])

    print("Standard deviation: ", average_accuracy, " +- ", standard_deviation)

    return best_tree



def run(training_dataset, test_dataset, max_share):

    x,y = training_dataset[:,:-1], [chr(training_dataset[i][-1]) for i in range(len(training_dataset))]
    x_test, y_test = test_dataset[:,:-1], [chr(test_dataset[i][-1]) for i in range(len(test_dataset))]

    # print("Training the decision tree...")
    classifier = DecisionTreeClassifier()
    classifier.set_max_share_hyperparameter(max_share)
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

    return accuracy, classifier



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