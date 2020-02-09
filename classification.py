import numpy as np
import helpers as hp
import Node as nd
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
        self.init_freq = {}
        self.max_share_hyperparameter = 0.00
    
    def set_max_share_hyperparameter(self, value):
        self.max_share_hyperparameter = value

    def induce_decision_tree(self, dataset):
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
        current_shares = hp.get_probabilities(hp.get_frequency(dataset), self.init_freq)
        max_share = max(current_shares.values()) 

        best_split = hp.find_best_split(dataset)
        # no more info gain or hyperparameter is set to pre-prune the tree
        if (best_split.attribute == None or max_share < self.max_share_hyperparameter): 
            node = nd.LeafNode(hp.get_frequency(dataset), self.init_freq)
        else :
            true_data, false_data = hp.split(dataset, best_split)
            child_true = self.induce_decision_tree(true_data)
            child_false = self.induce_decision_tree(false_data)
            node = nd.DecisionNode(best_split, child_true, child_false)
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
        
        dataset = hp.get_data(x, y)
        self.init_freq = hp.get_frequency(dataset)
        self.tree = self.induce_decision_tree(dataset)
        
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
        predictions = np.zeros((x.shape[0],), dtype=np.object)
        
        for i in range(len(x)):
            predictions[i] = chr(self.tree.question(x[i]))
    
        return predictions