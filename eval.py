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
    
    def confusion_matrix(self, prediction, annotation, class_labels=[]):
        """ Computes the confusion matrix.
        
        Parameters
        ----------
        prediction : np.array
            an N dimensional numpy array containing the predicted
            class labels
        annotation : np.array
            an N dimensional numpy array containing the ground truth
            class labels
        class_labels : np.array
            a C dimensional numpy array containing the ordered set of class
            labels. If not provided, defaults to all unique values in
            annotation.
        
        Returns
        -------
        np.array
            a C by C matrix, where C is the number of classes.
            Classes should be ordered by class_labels.
            Rows are ground truth per class, columns are predictions.
        """
        
        if (len(class_labels) == 0):
            class_labels = np.unique(annotation)
        
        confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)
        
            
        for i in range(len(prediction)):
            annotation_index = np.where(class_labels == annotation[i])
            prediction_index = np.where(class_labels == prediction[i])
            if(len(annotation_index[0]) > 0 and len(prediction_index[0]) > 0):
                row_num = annotation_index[0][0]
                col_num = prediction_index[0][0]
                confusion[row_num][col_num] += 1
        # TODO: throw error if prediction is not found in the class_labels?
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
        accuracy = 0.0
        true_total = 0
        total = 0
        for i in range (len(confusion)):
            true_total += confusion[i][i]
            for j in range(len(confusion)):
                total += confusion[i][j]
        if total > 0:
            raw_accuracy = true_total / total
            # accuracy = round(raw_accuracy, 1)

        return raw_accuracy

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
        precision = np.zeros((len(confusion), ))

        for i in range (len(confusion)):
            total_predicted_positive = 0
            true_positive = confusion[i][i]
            for j in range (len(confusion)):
                total_predicted_positive += confusion[j][i]
            if total_predicted_positive > 0:
                precision[i] = true_positive / total_predicted_positive
    
        macro_precision = 0
        for i in range (len(precision)):
            macro_precision += precision[i]
        macro_precision = macro_precision / len(precision)

        return (precision, macro_precision)
    
    
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
        recall = np.zeros((len(confusion), ))
        
        for i in range(len(confusion)):
            total_actual_positive = 0
            for j in range(len(confusion)):
                total_actual_positive += confusion[i][j]
            if total_actual_positive > 0: 
                recall[i] = confusion[i][i] / total_actual_positive
        
        # You will also need to change this        
        macro_recall = 0
        for i in range(len(recall)):
            macro_recall += recall[i]
        macro_recall = macro_recall / len(recall)

        return (recall, macro_recall)
    
    
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
        f1 = np.zeros((len(confusion), ))
        
        recall, _ = self.recall(confusion)
        precision, _ = self.precision(confusion)
        f1 = np.multiply(2, np.divide(np.multiply(recall, precision), np.add(recall, precision)))
        
        # You will also need to change this        
        macro_f1 = 0
        for i in range(len(confusion)):
            macro_f1 += f1[i]
        macro_f1 = macro_f1 / len(confusion)
        
        return (f1, macro_f1)
   
 
# Test
train_attributes, train_labels = hp.read_file("data/train_full.txt")
dtClassifier = cf.DecisionTreeClassifier()
dtClassifier.train(train_attributes, train_labels)

test_attributes, test_labels = hp.read_file("data/test.txt")
predictions = dtClassifier.predict(test_attributes)

evaluator = Evaluator()
confusion = evaluator.confusion_matrix(predictions, test_labels)
print(confusion)

print(evaluator.accuracy(confusion))
print(evaluator.precision(confusion))
print(evaluator.recall(confusion))
print(evaluator.f1_score(confusion))
