import numpy as np 
from question1 import *
from operator import itemgetter

def convertToAscii(label):
    newArray = []
    for i in range (0, len(label)):
        newArray.append(ord(label[i]))
    return newArray

def mergeArrays(label, attributes):
    mergedArr = []
    for i in range (0, len(label)):
        mergedArr.append(np.insert(attributes[i], 0, label[i]))
    return np.asarray(mergedArr)

def sortByColAndLabel(data, col):
        # print(col)
        sortedList = sorted(data, key=itemgetter(col, 0))
        sortedArr = np.asarray(sortedList)
        # print(sortedArr)
        # print("xxxxxxx")
        return sortedArr

### Hardcoded data ###
label, attributes = readFile("data/toy.txt")
######################

def findBestSplitPoint(data):
    bestIG = 0
    bestSplitPoint = None
    dataSet = mergeArrays(convertToAscii(label), attributes)

    # assuming label at position 0
    for i in range (1, len(dataSet[0])):
        sortedArr = sortByColAndLabel(dataSet, i)
        print(sortedArr)
        # find split points
        prevSplitPoint = None
        prevClass = sortedArr[0][0]
        hasMultipleClasses = False
        # start checking from 1st value because splitting at 0th index will return the original array
        for j in range (1, len(sortedArr)):
            classChanged = False
            # still need to skip some split points if useless (cf specs)    
            splitPoint = sortedArr[j][i]
            currClass = sortedArr[j][0]
            if (not hasMultipleClasses and prevClass != currClass):
                hasMultipleClasses = True
                classChanged = True
            if (hasMultipleClasses):
                if (classChanged):
                    print(splitPoint, currClass)
                elif (prevSplitPoint != splitPoint):
                    # check IG
                    print(splitPoint, currClass)
            prevSplitPoint = splitPoint
            prevClass = currClass

   


        ### WARNING ###
        ### by here, the current split point is [index, setOfAttributes] ###
        ### END OF WARNING ###
        
        # split in 2 subsets - use Ken's function
        # subset1, subset2 = []
        # for row in data:
        #     if row[setOfAttributes] <= data[index][setOfAttributes]:
        #         subset1.append(row[setOfAttributes])
        #     else :
        #         subset2.append(row[setOfAttributes])
        
        # subset1, subset1 = np.asarray(subset1), np.asarray(subset2)

        # # compare IG and choose the best one
        # tmpIG = calcIG(data, subset1, subset2)
        # if tmpIG > bestIG:
        #     bestIG = tmpIG
        #     bestSplitPoint = [index, setOfAttributes]
        
    return bestSplitPoint

findBestSplitPoint(readFile("data/toy.txt")[1])

