import numpy as np
import SplitInfo as si

# stud functions
def sortByColAndLabel(data, col):
    return np.array([[1, 2, 3], 
                     [4, 5, 6],
                     [7, 8, 9]])

def findBestSplitPoint(dataset):
    splitInfo = si.SplitInfo(1, 1)
    splitIndex = 2
    return splitInfo, splitIndex


# @param dataset NxAttr array
# @param splitInfo = [splitAttribute, splitPoint]
# @retrun an splitPoint x Attr array and an (N - splitPoint) x Attr array
def split(dataset, splitInfo):
    trueData = np.array([])
    falseData = np.array([])

    for data in dataset:
        if (data[0] > splitInfo.value):
            


# Tests
# dataset = np.array([[1, 2, 3],
#                     [4, 5, 6],
#                     [7, 8, 9]])
# splitInfo = [2, 2]
# print(split(dataset, splitInfo))