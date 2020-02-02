import numpy as np

# own libraries
import helpers as hp
import node as nd
import classification as cf
import eval as ev
import copy as cp

def prune(tree, node, validation, annotation, prevNode = None, left = None):

    if isinstance(node, nd.LeafNode):
        return True

    if isinstance(node, nd.DecisionNode):
        
        trueBranch = prune(tree, node.childTrue, validation, annotation, node, True)
        falseBranch = prune(tree, node.childFalse, validation, annotation, node, False) 
        if trueBranch and falseBranch:
            # print(node, 'with', node.childTrue, node.childFalse)

            basePrediction = tree.predict(validation)
            evaluator = ev.Evaluator()
            baseConfusion = evaluator.confusion_matrix(basePrediction, annotation)
            baseAccuracy = evaluator.accuracy(baseConfusion)

            dataset = np.append(node.childTrue.dataset, node.childFalse.dataset, axis=0)
            freq = node.childTrue.freq
            # print(freq)
            # print(prevNode, 'before reassign', prevNode.childTrue, prevNode.childFalse)
            if left:
                saved = cp.deepcopy(prevNode.childTrue)
                prevNode.childTrue = nd.LeafNode(dataset, freq)
            elif not left:
                saved = cp.deepcopy(prevNode.childFalse)
                prevNode.childFalse = nd.LeafNode(dataset, freq)

            # print(prevNode, 'after reassign', prevNode.childTrue, prevNode.childFalse)

            curPrediction = tree.predict(validation)
            curConfusion = evaluator.confusion_matrix(curPrediction, annotation)
            curAccuracy = evaluator.accuracy(curConfusion)
            # print(baseAccuracy, curAccuracy)

            if curAccuracy > baseAccuracy:
                print('returning true')
                return True
            else:
                # print('returning false')
                if left:
                    prevNode.childTrue = saved 
                elif not left:
                    prevNode.childFalse = saved 
                return False

    # print('getting to the end')
    return False

# test

train_attributes, train_labels = hp.readFile("data/train_sub.txt")
dtClassifier = cf.DecisionTreeClassifier()
dtClassifier.train(train_attributes, train_labels)

val_attributes, val_labels = hp.readFile("data/validation.txt")
# val_attributes = np.append(val_attributes, train_attributes, axis=0)
# val_labels = np.append(val_labels, train_labels, axis=0)

test_attributes, test_labels = hp.readFile("data/test.txt")

predictions = dtClassifier.predict(test_attributes)
evaluator = ev.Evaluator()
confusion = evaluator.confusion_matrix(predictions, test_labels)
print(evaluator.accuracy(confusion))
print(evaluator.precision(confusion)[-1])
print(evaluator.recall(confusion)[-1])
print(evaluator.f1_score(confusion)[-1])

prune(dtClassifier, dtClassifier.tree, val_attributes, val_labels)

predictions = dtClassifier.predict(test_attributes)
confusion = evaluator.confusion_matrix(predictions, test_labels)
print(evaluator.accuracy(confusion))
print(evaluator.precision(confusion)[-1])
print(evaluator.recall(confusion)[-1])
print(evaluator.f1_score(confusion)[-1])
