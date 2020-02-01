import numpy as np 
import helpers as hp
def read_file(filename):
    raw_data = open(filename).readlines() # a list of lines

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
    attributes = np.asarray([line[:-1] for line in data_list], int) # int() automatically removes blank spaces
    label = np.asarray([line[-1].strip() for line in data_list]) # use strip() to remove blank spaces
    # attributes - a [data_num X attribute_num] array
    # label -  a [data_num X 1] array

    # attributes_max = np.amax(attributes, axis = 0)
    # attributes_min = np.amin(attributes, axis = 0)
    # attributes_range = [(attributes_min[i], attributes_max[i]) for i in range(attribute_num)]


    # print("Number of data is " + str(data_num))
    # print("Number of attributes is " + str(attribute_num))
    # print("\nRange of attributes are ")
    # for i in range(attribute_num): 
    #     print("Attr {} : {}.".format(i, attributes_range[i]))
    # print("\nFrequency of labels are ")
    # for key, value in label_freq.items():
    #     print("Label '{}' : {}".format(key, value))

    return label, attributes


# print("\n********** Information of train_full.txt **********")
# read_file("data/train_full.txt")
# print("***********************************************\n")

# print("\n********** Information of tran_sub.txt **********")
# read_file("data/train_sub.txt")
# print("***********************************************\n")

# print("\n********** Information of tran_noisy.txt **********")
# read_file("data/train_noisy.txt")
# print("***********************************************\n")

# print("\n********** Information of validation.txt **********")
# read_file("data/validation.txt")
# print("***********************************************\n")

# print("\n********** Information of simple1.txt **********")
# read_file("data/simple1.txt")
# print("***********************************************\n")

# print("\n********** Information of simple2.txt **********")
# read_file("data/simple2.txt")
# print("***********************************************\n")

# print("\n********** Information of test.txt **********")
# read_file("data/test.txt")
# print("***********************************************\n")

# print("\n********** Information of toy.txt **********")
# read_file("data/toy.txt")
# print("***********************************************\n")