import SplitInfo as si
import helpers as hp

class Node:
    
    def print(self):
        raise NotImplementedError

    def question(self, attributes):
        raise NotImplementedError


class DecisionNode(Node):
    """
    A decision node
    
    Attributes
    ----------
    entropy : float
        Saves entropy for comparison with children entropy
    
    Methods
    -------
    
    calcChildEntropy(self, subLabel1, subLabel2) : float
        Given a split, compare the sum of the entropies from subLabel1 and subLabel2 with parent entropy

    """

    # split_info = si.SplitInfo(None, None)
    # child_true = Node()
    # child_false = Node()

    def __init__(self, _split_info, child_true, child_false):
        super().__init__()
        self.split_info = _split_info
        self.child_true = child_true
        self.child_false = child_false

        # Before pruning, there is no label stored in the Decision nodes but after pruning, the 
        # decision node is effectively a LeafNode

    def print(self, layer, json_data):
        for _ in range(layer):
            print('--', end='')
        print("Decision Node: Attribute {} is smaller than {}?"
            .format(self.split_info.attribute, self.split_info.value))
        json_data.append({
            "split_point": [self.split_info.attribute, int(self.split_info.value)],
            "child_true": [],
            "child_false": []
        })
        self.child_true.print(layer + 1, json_data[0]['child_true'])
        self.child_false.print(layer + 1, json_data[0]['child_false'])


    def question(self, attributes):        
        if (int(attributes[self.split_info.attribute]) < int(self.split_info.value)):
            return self.child_true.question(attributes)
        else:
            return self.child_false.question(attributes)

class LeafNode(Node):

    def __init__(self, dataset):
        super().__init__()
        self.predictions = hp.get_frequency(dataset)
        self.label = hp.get_major_label(self.predictions)

    def print(self, layer, json_data):
        for _ in range(layer):
            print('--', end='')
        json_data.append({
            "label": chr(self.label)
        })        
        print("Leaf Node: Label is {}".format(chr(self.label)))

    def question(self, attributes):
        return self.label