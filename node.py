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

    # splitInfo = si.SplitInfo(None, None)
    # childTrue = Node()
    # childFalse = Node()

    def __init__(self, _splitInfo, childTrue, childFalse):
        super().__init__()
        self.splitInfo = _splitInfo
        self.childTrue = childTrue
        self.childFalse = childFalse

        # Before pruning, there is no label stored in the Decision nodes but after pruning, the 
        # decision node is effectively a LeafNode

    def print(self, layer, jsonData):
        for _ in range(layer):
            print('--', end='')
        print("Decision Node: Attribute {} is smaller than {}?"
            .format(self.splitInfo.attribute, self.splitInfo.value))
        jsonData.append({
            "splitPoint": [self.splitInfo.attribute, int(self.splitInfo.value)],
            "childTrue": [],
            "childFalse": []
        })
        self.childTrue.print(layer + 1, jsonData[0]['childTrue'])
        self.childFalse.print(layer + 1, jsonData[0]['childFalse'])


    def question(self, attributes):
        if (attributes[self.splitInfo.attribute] < self.splitInfo.value):
            return self.childTrue.question(attributes)
        else:
            return self.childFalse.question(attributes)
        


        
# attributes, labels = readFile("data/toy.txt")

# root = DecisionNode(attributes, labels)
# print("Root has entropy: " + str(root.entropy))

# attributes1, labels1 = readFile("data/toy_sub1.txt")
# attributes2, labels2 = readFile("data/toy_sub2.txt")
# test = root.calcIG(labels1, labels2)
# print(test)


class LeafNode(Node):

    def __init__(self, dataset):
        super().__init__()
        self.predictions = hp.getFrequency(dataset)
        self.label = hp.getMajorLabel(self.predictions)

    def print(self, layer, jsonData):
        for _ in range(layer):
            print('--', end='')
        jsonData.append({
            "label": chr(self.label)
        })        
        print("Leaf Node: Label is {}".format(chr(self.label)))

    def question(self, attributes):
        return self.label