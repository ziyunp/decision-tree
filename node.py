from question1 import readFile
from question1 import countFrequency
import math

def calcEntropy(labels):
    freq = countFrequency(labels)
    total = 0
    for item in freq:
        total += freq[item] 

    entropy = 0

    for item in freq:
        entropy -= (freq[item] / total) * math.log(freq[item] / total, 2)

    return entropy

class Node:
    def __init__(self):
        self.entropy = 0


# potentially add:
    # pointers to sub nodes 
    # splittingRule - what's the format of this
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

    def __init__(self, attributes, labels):
        super().__init__()
        self.entropy = calcEntropy(labels)
        self.attributes = attributes
        self.labels = labels
        

    # splitPoint
    # labels
    # subLabel1
    # subLabel2



    def calcIG(self, subLabel1, subLabel2):
        dataCount = len(subLabel1) + len(subLabel2)
        childEntropy = (len(subLabel1) / dataCount) * calcEntropy(subLabel1) + (len(subLabel2) / dataCount) * calcEntropy(subLabel2)
        return self.entropy - childEntropy




        
attributes, labels = readFile("data/toy.txt")

root = DecisionNode(attributes, labels)
print("Root has entropy: " + str(root.entropy))

attributes1, labels1 = readFile("data/toy_sub1.txt")
attributes2, labels2 = readFile("data/toy_sub2.txt")
test = root.calcIG(labels1, labels2)
print(test)



