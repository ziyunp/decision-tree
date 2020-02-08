class SplitInfo:
    def __init__(self, _attribute, _value):
        self.attribute = _attribute
        self.value = _value

    def match(self, row):
        return row[self.attribute] < self.value