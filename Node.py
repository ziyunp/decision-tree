import SplitInfo as si
import helpers as hp

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


class Decision_node(Node):
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

    def print(self, layer, json_data):
        if (layer <= 10):
            for _ in range(layer):
                print('    ', end='')
            print("+---- ", end='')
            print("Decision Node: Attribute {} < {}? (entropy: {})"
                .format(self.split_info.attribute + 1, self.split_info.value, '%.3f'%(self.get_entropy())))
            json_data.append({
                "split_point": [self.split_info.attribute, int(self.split_info.value)],
                "entropy": self.get_entropy(),
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
    
    def get_cur_freq(self):
        ret_freq = hp.merge_freq(self.child_true.get_cur_freq(), self.child_false.get_cur_freq())
        return ret_freq

    def get_entropy(self):
        cur_freq = self.get_cur_freq()
        return hp.calc_entropy(cur_freq)

    def get_depth(self, cur_depth):
        return max(self.child_true.get_depth(cur_depth + 1), self.child_false.get_depth(cur_depth + 1))

class Leaf_node(Node):

    def __init__(self, cur_freq, init_freq):
        super().__init__()
        self.cur_freq = cur_freq
        self.init_freq = init_freq
        self.predictions = hp.get_probabilities(self.cur_freq, self.init_freq)
        self.label = hp.get_major_label(self.predictions)

    def print(self, layer, json_data):
        if (layer <= 10):
            for _ in range(layer):
                print('    ', end='')
            print("+---- ", end='')
            json_data.append({
                "label": chr(self.label),
                "entropy": self.get_entropy()
            })        
            print("Leaf {} (entropy: {})".format(chr(self.label), '%.3f'%(self.get_entropy())))

    def question(self, attributes):
        return self.label

    def get_cur_freq(self):
        return self.cur_freq

    def get_entropy(self):
        return hp.calc_entropy(self.cur_freq)

    def get_depth(self, cur_depth):
        return cur_depth