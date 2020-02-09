import numpy as np
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
        if (not isinstance(node, nd.LeafNode)):
            cur_freq, init_freq = remove_children(node)
            new_node = nd.LeafNode(cur_freq, init_freq)
            return new_node
    else:
        if (not isinstance(node, nd.LeafNode)):
            node.child_true = pruning_helper(node.child_true, max_depth - 1)
            node.child_false = pruning_helper(node.child_false, max_depth - 1)
    return node


def prune_more(dt_classifier,
               node,
               validation,
               annotation,
               prev_node=None,
               node_class=None):
    if isinstance(node, nd.LeafNode):
        return node
    else:
        if isinstance(node.child_true, nd.DecisionNode):
            node.child_true = prune_more(dt_classifier, node.child_true,
                                         validation, annotation, node, True)
        if isinstance(node.child_false, nd.DecisionNode):
            node.child_false = prune_more(dt_classifier, node.child_false,
                                          validation, annotation, node, False)

        # save the current node
        node_backup = cp.deepcopy(node)

        # get original accuracy
        base_prediction = dt_classifier.predict(validation)
        evaluator = ev.Evaluator()
        base_confusion = evaluator.confusion_matrix(base_prediction,
                                                    annotation)
        base_accuracy = evaluator.accuracy(base_confusion)

        # root_node
        if node_class == None:
            return node
        # prune chidlren
        cur_freq, init_freq = remove_children(node)
        node = nd.LeafNode(cur_freq, init_freq)
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
            return node
        return node_backup


def remove_children(node):
    if (isinstance(node.child_true, nd.LeafNode)):
        freq_true = node.child_true.predictions
        init_freq = node.child_true.init_freq
    else:
        freq_true, init_freq = remove_children(node.child_true)
    if (isinstance(node.child_false, nd.LeafNode)):
        freq_false = node.child_false.predictions
    else:
        freq_false, _ = remove_children(node.child_false)
    return hp.merge_freq(freq_true, freq_false), init_freq


def prune(tree, node, validation, annotation, prev_node=None, left=None):

    if isinstance(node, nd.LeafNode):
        return True

    if isinstance(node, nd.DecisionNode):

        true_branch = prune(tree, node.child_true, validation, annotation,
                            node, True)
        false_branch = prune(tree, node.child_false, validation, annotation,
                             node, False)
        if true_branch and false_branch:

            base_prediction = tree.predict(validation)
            evaluator = ev.Evaluator()
            base_confusion = evaluator.confusion_matrix(
                base_prediction, annotation)
            base_accuracy = evaluator.accuracy(base_confusion)

            cur_freq = hp.merge_freq(node.child_true.cur_freq,
                                     node.child_false.cur_freq)
            init_freq = node.child_true.init_freq

            if left:
                saved = cp.deepcopy(prev_node.child_true)
                prev_node.child_true = nd.LeafNode(cur_freq, init_freq)
            elif not left:
                saved = cp.deepcopy(prev_node.child_false)
                prev_node.child_false = nd.LeafNode(cur_freq, init_freq)

            cur_prediction = tree.predict(validation)
            cur_confusion = evaluator.confusion_matrix(cur_prediction,
                                                       annotation)
            cur_accuracy = evaluator.accuracy(cur_confusion)

            if cur_accuracy >= base_accuracy:
                return True
            if left:
                prev_node.child_true = saved
            elif not left:
                prev_node.child_false = saved
            return False

    return False
