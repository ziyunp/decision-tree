import SplitInfo as si
import helpers as hp

MAX_PRINT_DEPTH = 99


class Node:
    def print(self):
        raise NotImplementedError

    def question(self, attributes):
        raise NotImplementedError

    def get_entropy(self):
        raise NotImplementedError

    def get_cur_freq(self):
        raise NotImplementedError

    def get_depth(self, cur_depth):
        raise NotImplementedError


class DecisionNode(Node):
    """  
    Attributes
    ----------
    split_info: SplitInfo
        Saves the split point used at this node
    child_true: Node
        Stores the "true" subset that matches the split point
    child_false: Nodemaximum depth of the decision tree
        Recursively prints the tree at each layer
        Allow saving of the tree into a json file
    
    Methods
    -------
    question(self, attributes): 
        Recursively call the corresponding child's question method
        The child is selected by the attribute value used to split at this node
    
    get_cur_freq(self):
        Returns the frequency of class labels in the dataset at this node
    
    get_entropy(self):
        Returns the entropy at this node
    
    get_depth(self, cur_depth):
        Recursively called to return the maximum depth of the decision tree
    """
    def __init__(self, _split_info, child_true, child_false):
        super().__init__()
        self.split_info = _split_info
        self.child_true = child_true
        self.child_false = child_false

    def print(self, layer, json_data):
        if (layer <= MAX_PRINT_DEPTH):
            for _ in range(layer):
                print('    ', end='')
            print("+---- ", end='')
            print("Decision Node: Attribute {} < {}? (entropy: {})".format(
                self.split_info.attribute + 1, self.split_info.value,
                '%.3f' % (self.get_entropy())))
            json_data.append({
                "split_point":
                [self.split_info.attribute,
                 int(self.split_info.value)],
                "entropy":
                self.get_entropy(),
                "child_true": [],
                "child_false": []
            })
            self.child_true.print(layer + 1, json_data[0]['child_true'])
            self.child_false.print(layer + 1, json_data[0]['child_false'])

    def question(self, attributes):
        if (int(attributes[self.split_info.attribute]) < int(
                self.split_info.value)):
            return self.child_true.question(attributes)
        return self.child_false.question(attributes)

    def get_cur_freq(self):
        return hp.merge_freq(self.child_true.get_cur_freq(),
                             self.child_false.get_cur_freq())

    def get_entropy(self):
        return hp.calc_entropy(self.get_cur_freq())

    def get_depth(self, cur_depth):
        return max(self.child_true.get_depth(cur_depth + 1),
                   self.child_false.get_depth(cur_depth + 1))


class LeafNode(Node):
    """  
    Attributes
    ----------
    cur_freq: dictionary
        Stores the frequency of each class in this node
    init_freq: dictionary
        Stores the initial frequency of each class in the original dataset
    predictions: dictionary
        Stores the proportion of each class in this node of the original sample
    label: int
        Stores the majority class label in the form of ASCII code


    Methods
    -------
    print(self, layer, json_data): 
        Prints out the leaf node
        Allow saving of the tree into a json file
    
    question(self, attributes): 
        Returns self.label
    
    get_cur_freq(self):
        Returns self.cur_freq
    
    get_entropy(self):
        Returns the entropy at this node
    
    get_depth(self, cur_depth):
        Returns the value of cur_depth
    """
    def __init__(self, cur_freq, init_freq):
        super().__init__()
        self.cur_freq = cur_freq
        self.init_freq = init_freq
        self.predictions = hp.get_probabilities(self.cur_freq, self.init_freq)
        self.label = hp.get_major_label(self.predictions)

    def print(self, layer, json_data):
        if (layer <= MAX_PRINT_DEPTH):
            for _ in range(layer):
                print('    ', end='')
            print("+---- ", end='')
            json_data.append({
                "label": chr(self.label),
                "entropy": self.get_entropy()
            })
            print("Leaf {} (entropy: {})".format(chr(self.label), '%.3f' %
                                                 (self.get_entropy())))

    def question(self, attributes):
        return self.label

    def get_cur_freq(self):
        return self.cur_freq

    def get_entropy(self):
        return hp.calc_entropy(self.cur_freq)

    def get_depth(self, cur_depth):
        return cur_depth
