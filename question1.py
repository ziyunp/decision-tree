import numpy as np 
import operator

def readfile(filename):

    attributes = []
    labels = []
    ascii_labels = []

    with open(filename) as f:
        content = f.readlines()

    for line in content:
        line_list = line.split(",")
        label = line_list[-1].strip()
        for i in range (0, len(line_list) - 1):
            line_list[i] = int(line_list[i])
        attributes.append(line_list[0:-1])
        labels.append(label)
        # report Q1.1
        ascii_labels.append(ord(label))
    
    # print(np.array(attributes))
    # print(np.array(labels))

    # for i in range (0, len(attributes[0]) - 1):
        # print("#####################")
        # print(i, " column, min: ", np.amin(np.array(attributes[:][i])))
        # print(i, "column, max: ", np.amax(np.array(attributes[:][i])))
        # print("#####################")

    # print("label, min: ", chr(np.amin(np.array(ascii_labels))))
    # print("label, max: ", chr(np.amax(np.array(ascii_labels))))
    
    # report Q1.2
    labels_count = {}

    for label in labels:
        if label not in labels_count:
            labels_count[label] = 0
        labels_count[label] += 1

    print(labels_count)

    return np.array(attributes), np.array(labels)

readfile("data/train_sub.txt")
