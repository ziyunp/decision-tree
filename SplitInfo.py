class SplitInfo:
    """  
    Attributes
    ----------
    attribute: int
        Stores the index of the attribute of a split point
    value: int
        Stores the value of the attribute of a split point

    Methods
    -------
    match(self, instance): 
        Parameter: instance contains a list of values of different attributes
        The targeted attribute is at the same index as 'attribute'
        Returns True if the given attribute value is less than the split point's attribute value
    """

    def __init__(self, _attribute, _value):
        self.attribute = _attribute
        self.value = _value

    def match(self, instance):
        return instance[self.attribute] < self.value