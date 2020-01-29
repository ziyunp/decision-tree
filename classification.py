##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train() and predict() methods of the 
# DecisionTreeClassifier 
##############################################################################

import numpy as np

# own libraries
import helpers as hp
import node as nd
import json


class DecisionTreeClassifier(object):
    """
    A decision tree classifier
    
    Attributes
    ----------
    is_trained : bool
        Keeps track of whether the classifier has been trained
    
    Methods
    -------
    train(X, y)
        Constructs a decision tree from data X and label y
    predict(X)
        Predicts the class label of samples X
    
    """

    def __init__(self):
        self.is_trained = False
    

    def induceDecisionTree(self, dataset):
        """ Recursively inducing a decision tree
        
        Parameters
        ----------
        dataset : numpy.array
            An N by K+1 numpy array (N is the number of instances, K is the 
            number of attributes and the first column is labels)
        
        Returns
        -------
        Node
            The root node of the induced decision tree
        
        """
        # recursively inducing a decision tree
        bestSplit = hp.findBestSplitPoint(dataset)
        if (bestSplit.attribute == None): # all samples have the same label or cannot be split
            node = nd.LeafNode(dataset)
        else :
            trueData, falseData = hp.split(dataset, bestSplit)
            childTrue = self.induceDecisionTree(trueData)
            childFalse = self.induceDecisionTree(falseData)
            node = nd.DecisionNode(bestSplit, childTrue, childFalse)

        return node 
    
    def train(self, x, y):
        """ Constructs a decision tree classifier from data
        
        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of instances, K is the 
            number of attributes)
        y : numpy.array
            An N-dimensional numpy array
        
        Returns
        -------
        DecisionTreeClassifier
            A copy of the DecisionTreeClassifier instance
        
        """
        
        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."
        
        dataset = hp.getData(x, y)
        self.tree = self.induceDecisionTree(dataset)
        
        # set a flag so that we know that the classifier has been trained
        self.is_trained = True
        
        return self
    
    
    def predict(self, x):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.
        
        Assumes that the DecisionTreeClassifier has already been trained.
        
        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of samples, K is the 
            number of attributes)
        
        Returns
        -------
        numpy.array
            An N-dimensional numpy array containing the predicted class label
            for each instance in x
        """
        
        # make sure that classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("Decision Tree classifier has not yet been trained.")
        
        # set up empty N-dimensional vector to store predicted labels 
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=np.object)
        
        for i in range(len(x)):
            predictions[i] = self.tree.question(x[i])
    
        # remember to change this if you rename the variable
        return predictions
        


# Tests

### Hardcoded data ###
attributes, labels = hp.readFile("data/train_sub.txt")
jsonData = {}
jsonData["root"] = []
######################

dtClassifier = DecisionTreeClassifier()
dtClassifier.train(attributes, labels).tree.print(0, jsonData["root"])

test_attributes, test_labels = hp.readFile("data/test.txt")
predictions = dtClassifier.predict(test_attributes)

correct = 0
for i in range(len(test_labels)):
    if (ord(predictions[i]) == test_labels[i]):
        correct += 1
print(correct / len(test_labels))


with open('save.json', 'w') as outfile:
    json.dump(jsonData, outfile)