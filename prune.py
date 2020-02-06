import numpy as np

# own libraries
import helpers as hp
import Node as nd
import classification as cf
import eval as ev
import copy as cp

def prune_to_max_depth(dt_classifier, max_depth):
    dt_classifier.tree = pruning_helper(dt_classifier.tree, max_depth)
    return dt_classifier

def pruning_helper(node, max_depth):
    if (max_depth == 0):
        if (not isinstance(node, nd.Leaf_node)):
            cur_freq, init_freq = remove_children(node)
            new_node = nd.Leaf_node(cur_freq, init_freq)
            return new_node
        # else: # a Leaf Node, no need to change
    else: # a Decision Node, continue to ge deeper
        if (not isinstance(node, nd.Leaf_node)):
            node.childTrue = pruning_helper(node.childTrue, max_depth - 1)
            node.childFalse = pruning_helper(node.childFalse, max_depth - 1)
    return node


def prune_more(dt_classifier, node, validation, annotation, prev_node = None, node_class = None):
    if isinstance(node, nd.Leaf_node):
        return node
    else:
        if isinstance(node.child_true, nd.Decision_node):
            node.child_true = prune_more(dt_classifier, node.child_true, validation, annotation, node, True)
        if isinstance(node.child_false, nd.Decision_node):
            node.child_false = prune_more(dt_classifier, node.child_false, validation, annotation, node, False)

        # save the current node
        node_backup = cp.deepcopy(node)   
        
        # get original accuracy
        base_prediction = dt_classifier.predict(validation)
        evaluator = ev.Evaluator()
        base_confusion = evaluator.confusion_matrix(base_prediction, annotation)
        base_accuracy = evaluator.accuracy(base_confusion)

        # prune chidlren
        if node_class == None: # root_node
            return node
        cur_freq, init_freq = remove_children(node)
        node = nd.Leaf_node(cur_freq, init_freq)
        if node_class == True:
            prev_node.child_true = node
        elif node_class == False:
            prev_node.child_false = node

        # get new accuracy
        cur_prediction = dt_classifier.predict(validation)
        cur_confusion = evaluator.confusion_matrix(cur_prediction, annotation)
        cur_accuracy = evaluator.accuracy(cur_confusion)

        # decide whether confirm pruning
        if cur_accuracy >= base_accuracy:
            print("The accuracy improves from {} to {}".format(base_accuracy, cur_accuracy))
            return node
        else:
            return node_backup

def remove_children(node):
    if (isinstance(node.child_true, nd.Leaf_node)):
        freq_true = node.child_true.predictions
        init_freq = node.child_true.init_freq
    else:
        freq_true, init_freq = remove_children(node.child_true)
    if (isinstance(node.child_false, nd.Leaf_node)):
        freq_false = node.child_false.predictions
    else:
        freq_false, _ = remove_children(node.child_false)
    return hp.merge_freq(freq_true, freq_false), init_freq


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
