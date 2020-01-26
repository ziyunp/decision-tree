import numpy as np 

def read_file(filename):
    raw_data = open(filename).read().splitlines() # a list of lines

    # check whether there is data
    data_num = len(raw_data)
    if (data_num == 0):
        print("No data found in the given file")
        return
    
    # check whether attributes of each line is consistent
    attribute_num = len(raw_data[0].split(",")) - 1 
    # the last column is label
    for raw_line in raw_data:
        line = raw_line.split(",")
        if (attribute_num != len(line) - 1):
            print("Attribute numbers of each line is not consistent")
            return 

    data_list = [raw_line.split(',') for raw_line in raw_data]
    # a [data_num X (attribute_num + 1)] list
    attributes = [np.asarray(line[:-2], int) for line in data_list]
    label = [np.asarray(ord(line[-1]), int) for line in data_list]
    # attributes - a [data_num X attribute_num] array
    # label -  a [data_num X 1] array

    attributes_max = np.amax(attributes, axis = 0)
    attributes_min = np.amin(attributes, axis = 0)

    label_max = np.amax(label)
    label_min = np.amin(label)   

    print("Number of data is " + str(data_num))
    print("Number of attributs is " + str(attribute_num))
    print("Range of attributes are ")
    print(attributes_max)
    print(attributes_min)
    print("Range of labels are ")
    print(label_max)
    print(label_min)

    return attributes, label

read_file("data/train_full.txt")