import numpy as np 
from question1 import *
import math
import SplitInfo as si

def convertToAscii(label):
    newArray = []
    for i in range (0, len(label)):
        newArray.append(ord(label[i]))
    return newArray

def mergeArrays(label, attr):
    mergedArr = []
    for i in range (0, len(label)):
        mergedArr.append(np.insert(attr[i], 0, label[i]))
    return np.asarray(mergedArr)

def sortByAttrAndLabel(data, col):
        sortedList = sorted(data, key=lambda x:(x[col], x[0]))
        sortedArr = np.asarray(sortedList)
        return sortedArr

def calcEntropy(labels):
    freq = countFrequency(labels)
    total = 0
    for item in freq:
        total += freq[item] 

    entropy = 0

    for item in freq:
        entropy -= (freq[item] / total) * math.log(freq[item] / total, 2)

    return entropy

def calcIG(data, subLabel1, subLabel2):
    dataCount = len(subLabel1) + len(subLabel2)
    childEntropy = (len(subLabel1) / dataCount) * calcEntropy(subLabel1) + (len(subLabel2) / dataCount) * calcEntropy(subLabel2)
    return calcEntropy(data) - childEntropy

def checkIG(data, attr, splitPoint):
    # split in 2 subsets
    subset1 = []
    subset2 = []
    subset1, subset2 = split(data, attr, splitPoint)
    return calcIG(data[:,0], subset1[:,0], subset2[:,0])

def findBestSplitPoint(dataSet):
    bestIG = 0
    bestSplit = si.SplitInfo(None, None)

    # assuming label at position 0
    for attr in range (1, len(dataSet[0])):
        sortedArr = sortByAttrAndLabel(dataSet, attr)
        # print(sortedArr)

        # find split points
        prevSplitPoint = sortedArr[0][attr]
        # start checking from 1st value because splitting at 0th index will return the original array
        for row in range (1, len(sortedArr)):
            splitPoint = sortedArr[row][attr]
            currClass = sortedArr[row][0] # because first attr contains labels
            if ((prevSplitPoint != splitPoint)):
                # check IG
                currIG = checkIG(sortedArr, attr, splitPoint)
                if currIG > bestIG:
                    bestIG = currIG
                    bestSplit.attribute = attr
                    bestSplit.value = splitPoint
            prevSplitPoint = splitPoint

    return bestSplit

# @param dataset NxAttr array
# @param splitInfo = [splitAttribute, splitPoint]
# @retrun an splitPoint x Attr array and an (N - splitPoint) x Attr array
def split(dataset, attr, splitPoint):
    trueData = []
    falseData = []

    for data in dataset:
        if (data[attr] < splitPoint):
            trueData.append(data)  
        else: 
            falseData.append(data)
    return np.array(trueData), np.array(falseData)



### Hardcoded data ###
labels, attributes = readFile("data/toy.txt")
######################

dataSet = mergeArrays(convertToAscii(labels), attributes)
bestSplit = findBestSplitPoint(dataSet)
# print(bestSplit.attribute, bestSplit.value)

