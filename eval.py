##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: 
# Complete the following methods of Evaluator:
# - confusion_matrix()
# - accuracy()
# - precision()
# - recall()
# - f1_score()
##############################################################################

import numpy as np

import helpers as hp
import classification as cf





class Evaluator(object):
    """ Class to perform evaluation
    """
    
    def confusion_matrix(self, prediction, annotation, classLabels=[]):
        """ Computes the confusion matrix.
        
        Parameters
        ----------
        prediction : np.array
            an N dimensional numpy array containing the predicted
            class labels
        annotation : np.array
            an N dimensional numpy array containing the ground truth
            class labels
        classLabels : np.array
            a C dimensional numpy array containing the ordered set of class
            labels. If not provided, defaults to all unique values in
            annotation.
        
        Returns
        -------
        np.array
            a C by C matrix, where C is the number of classes.
            Classes should be ordered by classLabels.
            Rows are ground truth per class, columns are predictions.
        """
        
        if (len(classLabels) == 0):
            classLabels = np.unique(annotation)
        
        confusion = np.zeros((len(classLabels), len(classLabels)), dtype=np.int)
        
            
        for i in range(len(prediction)):
            rowNum = np.where(classLabels == annotation[i])[0][0]
            colNum = np.where(classLabels == prediction[i])[0][0]
            confusion[rowNum][colNum] += 1
        
        return confusion
    
    
    def accuracy(self, confusion):
        """ Computes the accuracy given a confusion matrix.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions
        
        Returns
        -------
        float
            The accuracy (between 0.0 to 1.0 inclusive)
        """

        trueTotal = 0
        total = 0
        for i in range (len(confusion)):
            trueTotal += confusion[i][i]
            for j in range(len(confusion)):
                total += confusion[i][j]

        rawAccuracy = trueTotal / total
        accuracy = round(rawAccuracy, 1)
        # print("accuracy: ", accuracy)
        return accuracy

    def precision(self, confusion):
        """ Computes the precision score per class given a confusion matrix.
        
        Also returns the macro-averaged precision across classes.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.
        
        Returns
        -------
        np.array
            A C-dimensional numpy array, with the precision score for each
            class in the same order as given in the confusion matrix.
        float
            The macro-averaged precision score across C classes.   
        """
        
        # Initialise array to store precision for C classes
        p = np.zeros((len(confusion), ))

        for i in range (0, len(confusion)):
            total = 0
            TP = confusion[i][i]
            for j in range (0, len(confusion)):
                total += confusion[j][i]
            p[i] = TP / total

        macro_p = 0
        for i in range (0, len(p)):
            macro_p += p[i]
        macro_p = macro_p / len(p)

        return (p, macro_p)
    
    
    def recall(self, confusion):
        """ Computes the recall score per class given a confusion matrix.
        
        Also returns the macro-averaged recall across classes.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.
        
        Returns
        -------
        np.array
            A C-dimensional numpy array, with the recall score for each
            class in the same order as given in the confusion matrix.
        
        float
            The macro-averaged recall score across C classes.   
        """
        
        # Initialise array to store recall for C classes
        r = np.zeros((len(confusion), ))
        
        for i in range(len(confusion)):
            for j in range(len(confusion)):
                r[i] += confusion[i][j]
            r[i] = confusion[i][i] / r[i]
        
        # You will also need to change this        
        macro_r = 0
        for i in range(len(r)):
            macro_r += r[i]
        macro_r = macro_r / len(r)

        return (r, macro_r)
    
    
    def f1_score(self, confusion):
        """ Computes the f1 score per class given a confusion matrix.
        
        Also returns the macro-averaged f1-score across classes.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.
        
        Returns
        -------
        np.array
            A C-dimensional numpy array, with the f1 score for each
            class in the same order as given in the confusion matrix.
        
        float
            The macro-averaged f1 score across C classes.   
        """
        
        # Initialise array to store recall for C classes
        f = np.zeros((len(confusion), ))
        
        recall, _ = self.recall(confusion)
        # precision, _ = self.precision(confusion)
        precision, _ = self.recall(confusion)
        f = np.multiply(2, np.divide(np.multiply(recall, precision), np.add(recall, precision)))
        
        # You will also need to change this        
        macro_f = 0
        for i in range(len(confusion)):
            macro_f += f[i]
        macro_f = macro_f / len(confusion)
        
        return (f, macro_f)
   
 
# # Test
# train_attributes, train_labels = hp.readFile("data/train_full.txt")
# dtClassifier = cf.DecisionTreeClassifier()
# dtClassifier.train(train_attributes, train_labels)

# test_attributes, test_labels = hp.readFile("data/test.txt")
# predictions = dtClassifier.predict(test_attributes)

# evaluator = Evaluator()
# confusion = evaluator.confusion_matrix(predictions, test_labels)
# print(confusion)

# print(evaluator.accuracy(confusion))
# print(evaluator.precision(confusion))
# print(evaluator.recall(confusion))
# print(evaluator.f1_score(confusion))