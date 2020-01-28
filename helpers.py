import numpy as np 
from question1 import *
from node import *

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
        # print(col)
        sortedList = sorted(data, key=lambda x:(x[col], x[0]))
        sortedArr = np.asarray(sortedList)
        # print(sortedArr)
        # print("xxxxxxx")
        return sortedArr

def checkIG(data, attr, splitPoint):
    
    # split in 2 subsets - use Ken's function
    subset1 = []
    subset2 = []
    for row in data:
        subset1.append(row) if (row[attr] < splitPoint) else subset2.append(row)
    subset1, subset2 = np.asarray(subset1), np.asarray(subset2)
    return calcIG(data, subset1, subset2)

### Hardcoded data ###
label, attributes = readFile("data/toy.txt")
######################

def findBestSplitPoint(data):
    # bestIG = checkIG(data, 0, 1)
    bestSplitPoint = None
    dataSet = mergeArrays(convertToAscii(label), attributes)

    # assuming label at position 0
    for attr in range (1, len(dataSet[0])):
        sortedArr = sortByAttrAndLabel(dataSet, attr)
        print(sortedArr)

        # find split points
        # prevClass = sortedArr[0][0]
        prevSplitPoint = sortedArr[0][attr]
        # start checking from 1st value because splitting at 0th index will return the original array
        for row in range (1, len(sortedArr)):
            # classChanged = False
            # still need to skip some split points if useless (cf specs)    
            splitPoint = sortedArr[row][attr]
            currClass = sortedArr[row][0] # because first attr contains labels
            if ((prevSplitPoint != splitPoint)):
                # check IG
                currIG = checkIG(sortedArr, attr, splitPoint)
                if currIG > bestIG:
                    bestIG = currIG
                    bestSplitPoint = [attr, splitPoint]
                # print(splitPoint, currClass)
            prevSplitPoint = splitPoint
            # prevClass = currClass


    return bestSplitPoint

findBestSplitPoint(readFile("data/toy.txt")[1])

