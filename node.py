import SplitInfo as si

class Node:
    def __init__(self):
        self.entropy = 0
    
    def print(self):
        raise NotImplementedError

    def question(self, attributes):
        raise NotImplementedError


# potentially add:
    # pointers to sub nodes 
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

    splitInfo = si.SplitInfo(None, None)
    childTrue = Node()
    childFalse = Node()

    def __init__(self, _splitInfo):
        super().__init__()
        self.splitInfo = _splitInfo
        # DecisionNode should not store entropy since entropy is related to a specific dataset. 

        # Before pruning, there is no label stored in the Decision nodes but after pruning, the 
        # decision node is effectively a LeafNode

        # self.entropy = calcEntropy(labels)
        # self.attributes = attributes
        # self.labels = labels


    def print(self, layer):
        for _ in range(layer):
            print('--', end='')
        print("Desion Node: Attribute {} is smaller than {}?"
            .format(self.splitInfo.attribute, self.splitInfo.value))
        self.childTrue.print(layer + 1)
        self.childFalse.print(layer + 1)


    def question(self, attributes):
        if (attributes[self.splitInfo.attribute - 1] < self.splitInfo.value):
            return self.childTrue.question(attributes)
        else:
            return self.childFalse.question(attributes)
        

    # splitPoint
    # labels
    # subLabel1
    # subLabel2



    # def calcIG(self, subLabel1, subLabel2):
    #     dataCount = len(subLabel1) + len(subLabel2)
    #     childEntropy = (len(subLabel1) / dataCount) * calcEntropy(subLabel1) + (len(subLabel2) / dataCount) * calcEntropy(subLabel2)
    #     return self.entropy - childEntropy




        
# attributes, labels = readFile("data/toy.txt")

# root = DecisionNode(attributes, labels)
# print("Root has entropy: " + str(root.entropy))

# attributes1, labels1 = readFile("data/toy_sub1.txt")
# attributes2, labels2 = readFile("data/toy_sub2.txt")
# test = root.calcIG(labels1, labels2)
# print(test)


class LeafNode(Node):

    def __init__(self, dataset):
        Node()
        self.label = dataset[0][0]

    def print(self, layer):
        for _ in range(layer):
            print('--', end='')        
        print("Leaf Node: Label is {}".format(chr(self.label)))

    def question(self, attributes):
        return chr(self.label)