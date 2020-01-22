import numpy as np 

def readfile(filename):

    attributes = []
    labels = []
    ascii_labels = []

    with open(filename) as f:
        content = f.readlines()

    min, max
    for line in content:
        line_list = line.split(",")
        for i in range (0, len(line_list) - 1):
            line_list[i] = int(line_list[i])
        attributes.append(line_list[0:-1])
        labels.append(line_list[-1].strip())
        for label in labels:
            ascii_labels.append(ord(label))
    
    # print(np.array(attributes))
    # print(np.array(labels))

    for i in range (0, len(attributes[0]) - 1):
        # print("#####################")
        print(i, " column, min: ", np.amin(np.array(attributes[:][i])))
        print(i, "column, max: ", np.amax(np.array(attributes[:][i])))
        # print("#####################")

    print("label, min: ", chr(np.amin(np.array(ascii_labels))))
    print("label, max: ", chr(np.amax(np.array(ascii_labels))))


    return np.array(attributes), np.array(labels)

readfile("data/train_full.txt")
