import question1

class Node:
    def __init__(self):
        self.entropy = 0

class DecisionNode(Node):

    """
    A decision node
    
    Attributes
    ----------
    entropy : float
        Saves entropy for comparison with children entropy
    
    Methods
    -------
    calcChildEntropy(subset1, subset2)

    """

    def __init__(self, labels):
        super.init()
        self.entropy = calcEntropy(labels)

    # splitPoint
    # labels
    # subset1
    # subset2

    def calcEntropy(labels):
        freq = countFrequency(labels)
        total = 0
        for item in freq:
            total += freq[item] 

        return total
        

attributes, labels = readFile("data/toy.txt")
test = calcEntropy(labels)
print(test)





# class LeafNode(Node):
#     # subset


