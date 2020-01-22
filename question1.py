import numpy as np 


attributes = []
labels = []

def readfile(filename):

    with open(filename) as f:
        content = f.readlines()

    for line in content:
        line_list = line.split(",")
        attributes.append(line_list[0:-1])
        labels.append(line_list[-1].strip())
    
    # print(np.array(attributes))
    # print(np.array(labels))

    return np.array(attributes), np.array(labels)

# readfile("data/simple1.txt")
