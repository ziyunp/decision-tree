import numpy as np 
from question1 import *
import math

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
    
    # split in 2 subsets - use Ken's function
    subset1 = []
    subset2 = []
    for row in data:
        subset1.append(row) if (row[attr] < splitPoint) else subset2.append(row)
    subset1, subset2 = np.asarray(subset1), np.asarray(subset2)
    return calcIG(data[:,0], subset1[:,0], subset2[:,0])

### Hardcoded data ###
label, attributes = readFile("data/toy.txt")
######################

def findBestSplitPoint(data):
    bestIG = 0
    bestSplitPoint = None
    dataSet = mergeArrays(convertToAscii(label), attributes)

    # assuming label at position 0
    for attr in range (1, len(dataSet[0])):
        sortedArr = sortByAttrAndLabel(dataSet, attr)
        print(sortedArr)

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
                    bestSplitPoint = [attr, splitPoint]
            prevSplitPoint = splitPoint

    return bestSplitPoint

# @param dataset NxAttr array
# @param splitInfo = [splitAttribute, splitPoint]
# @retrun an splitPoint x Attr array and an (N - splitPoint) x Attr array
def split(dataset, splitInfo):
    trueData = np.array([])
    falseData = np.array([])

    for data in dataset:
        if (data[0] > splitInfo.value):

findBestSplitPoint(readFile("data/toy.txt")[1])

