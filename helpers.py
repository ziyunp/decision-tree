import numpy as np 
import math
import SplitInfo as si


def readFile(filename):
    rawData = open(filename).read().splitlines() # a list of lines

    # check whether there is data
    dataNum = len(rawData)
    if (dataNum == 0):
        print("No data found in the given file")
        return

    # check whether attributes of each line is consistent
    attributeNum = len(rawData[0].split(",")) - 1 
    # the last column is label
    for rawLine in rawData:
        line = rawLine.split(",")
        if (attributeNum != len(line) - 1):
            print("Attribute numbers of each line is not consistent")
            return 

    data = [rawLine.split(',') for rawLine in rawData]
    attributes = np.asarray([data[row][:-1] for row in range(len(data))], int)
    labels = np.asarray([[ord(data[row][-1].strip())] for row in range(len(data))])

    return attributes, labels

def getData(attributes, labels):
    return np.append(attributes, labels, 1)



def getFrequency(dataset):
    freq = {}
    for item in dataset:
        if item[-1] not in freq:
            freq[item[-1]] = 0
        freq[item[-1]] += 1
    return freq

# def convertToAscii(label):
#     newArray = []
#     for i in range (0, len(label)):
#         newArray.append(ord(label[i]))
#     return newArray

# def mergeArrays(label, attr):
#     mergedArr = []
#     for i in range (0, len(label)):
#         mergedArr.append(np.insert(attr[i], 0, label[i]))
#     return np.asarray(mergedArr)


def sortByAttrAndLabel(data, col):
        sortedList = sorted(data, key=lambda x:(x[col], x[0]))
        sortedArr = np.asarray(sortedList)
        return sortedArr


def calcEntropy(dataset):
    freq = getFrequency(dataset)
    total = sum(freq.values())
    entropy = 0

    for item in freq:
        entropy -= (freq[item] / total) * math.log(freq[item] / total, 2)

    return entropy


def calcIG(baseEntropy, trueData, falseData):
    
    dataCount = len(trueData) + len(falseData)
    p = len(trueData) / dataCount
    childEntropy = p * calcEntropy(trueData) + (1-p) * calcEntropy(falseData)
    return baseEntropy - childEntropy

# def checkIG(data, attr, splitPoint):
#     # split in 2 subsets
#     subset1 = []
#     subset2 = []
#     subset1, subset2 = split(data, attr, splitPoint)
#     return calcIG(data[:,0], subset1[:,0], subset2[:,0])


# @param dataset NxAttr array
# @param splitInfo = [splitAttribute, splitPoint]
# @retrun an splitPoint x Attr array and an (N - splitPoint) x Attr array
def split(dataset, splitInfo):
    trueData = []
    falseData = []

    for data in dataset:
        if (splitInfo.match(data)):
            trueData.append(data)  
        else: 
            falseData.append(data)
    return np.array(trueData), np.array(falseData)

# dataset = getData("data/toy.txt")
# s = si.SplitInfo(0, 5)
# trueDataset, falseDataset = split(dataset, s)
# print(trueDataset)


def findBestSplitPoint(dataset):
    bestIG = 0
    bestSplit = si.SplitInfo(None, None)

    baseEntropy = calcEntropy(dataset)

    for attr in range (len(dataset[0]) - 1):
        sortedArr = sortByAttrAndLabel(dataset, attr)
        # print(sortedArr)

        # find split points
        prevSplitPoint = sortedArr[0][attr]
        # start checking from 1st value because splitting at 0th index will return the original array
        for row in range (1, len(sortedArr)):

            splitPoint = sortedArr[row][attr]

            if ((prevSplitPoint != splitPoint)):
                # check IG
                trueData, falseData = split(dataset, si.SplitInfo(attr, splitPoint))
                currIG = calcIG(baseEntropy, trueData, falseData)
                if currIG > bestIG:
                    bestIG = currIG
                    bestSplit = si.SplitInfo(attr, splitPoint)
            prevSplitPoint = splitPoint

    return bestSplit


### Hardcoded data ###
# dataset = getData("data/toy.txt")
######################

# bestSplit = findBestSplitPoint(dataset)
# print(bestSplit.attribute, bestSplit.value)


def getMajorLabel(probabilities):
    
    values = list(probabilities.values())
    keys = list(probabilities.keys())
    return keys[values.index(max(values))]


def getProbabilities(currentFrequency, initialFrequency):

    probabilities = {}
    total = sum(currentFrequency.values())

    for item in currentFrequency:
        probabilities[item] = currentFrequency[item] / (initialFrequency[item] if bool(initialFrequency) else total)

    return probabilities
