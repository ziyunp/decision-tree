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
import question1 as q1


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
    
    # recursively inducing a decision tree
    def induceDecisionTree(self, dataset):
        bestSplit = hp.findBestSplitPoint(dataset)
        if (bestSplit.attribute == None): # all samples have the same label or cannot be split
            node = nd.LeafNode(dataset)
        else :
            node = nd.DecisionNode(bestSplit)
            trueData, falseData = hp.split(dataset, bestSplit.attribute, bestSplit.value)
            node.childTrue = self.induceDecisionTree(trueData)
            node.childFalse = self.induceDecisionTree(falseData)
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
        
        dataSet = hp.mergeArrays(hp.convertToAscii(y), x)
        self.tree = self.induceDecisionTree(dataSet)
        
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
        
        
        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################
        
    
        # remember to change this if you rename the variable
        return predictions
        


# Tests

### Hardcoded data ###
labels, attributes = q1.readFile("data/toy.txt")
######################

dtClassifier = DecisionTreeClassifier()
dtClassifier.train(attributes, labels).tree.print(0)