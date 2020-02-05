import numpy as np

# own libraries
import helpers as hp
import Node as nd
import classification as cf
import eval as ev
import copy as cp

def prune(tree, node, validation, annotation, prev_node = None, left = None):

    if isinstance(node, nd.Leaf_node):
        return True

    if isinstance(node, nd.Decision_node):
        
        true_branch = prune(tree, node.child_true, validation, annotation, node, True)
        false_branch = prune(tree, node.child_false, validation, annotation, node, False) 
        if true_branch and false_branch:
            # print(node, 'with', node.child_true, node.child_false)

            base_prediction = tree.predict(validation)
            evaluator = ev.Evaluator()
            base_confusion = evaluator.confusion_matrix(base_prediction, annotation)
            base_accuracy = evaluator.accuracy(base_confusion)

            cur_freq = hp.merge_freq(node.child_true.cur_freq, node.child_false.cur_freq)
            init_freq = node.child_true.init_freq
            # print(freq)
            # print(prev_node, 'before reassign', prev_node.child_true, prev_node.child_false)
            if left:
                saved = cp.deepcopy(prev_node.child_true)
                prev_node.child_true = nd.Leaf_node(cur_freq, init_freq)
            elif not left:
                saved = cp.deepcopy(prev_node.child_false)
                prev_node.child_false = nd.Leaf_node(cur_freq, init_freq)

            # print(prev_node, 'after reassign', prev_node.child_true, prev_node.child_false)

            cur_prediction = tree.predict(validation)
            cur_confusion = evaluator.confusion_matrix(cur_prediction, annotation)
            cur_accuracy = evaluator.accuracy(cur_confusion)
            # print(base_accuracy, cur_accuracy)

            if cur_accuracy >= base_accuracy:
                # print('returning true')
                return True
            else:
                # print('returning false')
                if left:
                    prev_node.child_true = saved 
                elif not left:
                    prev_node.child_false = saved 
                return False

    # print('getting to the end')
    return False

# test

train_attributes, train_labels = hp.read_file("data/train_full.txt")
dtClassifier = cf.DecisionTreeClassifier()
dtClassifier.train(train_attributes, train_labels)

# print()
# print(dtClassifier.tree.get_cur_freq())
# print(dtClassifier.tree.get_entropy())

val_attributes, val_labels = hp.read_file("data/validation.txt")
# val_attributes = np.append(val_attributes, train_attributes, axis=0)
# val_labels = np.append(val_labels, train_labels, axis=0)

test_attributes, test_labels = hp.read_file("data/test.txt")

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

# print()
# print(dtClassifier.tree.get_cur_freq())
# print(dtClassifier.tree.get_entropy())

# print(hp.calc_entropy(hp.get_frequency(hp.get_data(train_attributes, train_labels))))