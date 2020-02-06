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
