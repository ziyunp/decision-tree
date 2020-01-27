import SplitInfo as si

class Node:
    def __init__(self):
        self.entropy = 0

class DecisionNode(Node):
    splitInfo = si.SplitInfo(-1, -1)
    childNodeL = Node()
    childNodeR = Node()

    def __init__(self):
        Node()

class LeafNode(Node):
    label = "A"
    
    def __init__(self):
        Node()