import numpy as np 
import math
import SplitInfo as si

LABEL_COL = -1

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

    data = [raw_line.split(',') for raw_line in raw_data]
    attributes = np.array([data[row][:-1] for row in range(len(data))], int)
    labels = np.array([data[row][-1].strip() for row in range(len(data))])
    return attributes, labels

def get_data(attributes, labels):
    labels = np.array([[ord(label)] for label in labels])
    return np.append(attributes, labels, axis=1)


def get_frequency(dataset):
    freq = {}
    for item in dataset:
        if item[LABEL_COL] not in freq:
            freq[item[LABEL_COL]] = 0
        freq[item[LABEL_COL]] += 1
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
#     return np.array(mergedArr)


def sort_by_attr_and_label(data, col):
        sortedList = sorted(data, key=lambda x:(x[col], x[-1]))
        sortedArr = np.array(sortedList)
        return sortedArr


def calc_entropy(freq):
    total = sum(freq.values())
    entropy = 0

    for item in freq:
        entropy -= (freq[item] / total) * math.log(freq[item] / total, 2)

    return entropy

def calc_info_gain(sorted_dataset, split_info):
    dataset_freq = get_frequency(sorted_dataset)
    base_entropy = calc_entropy(dataset_freq)
    data_count = len(sorted_dataset)

    for i in range (len(sorted_dataset)):
        if not split_info.match(sorted_dataset[i]):
            split_index = i
            break

    p = split_index / data_count
    true_freq = get_frequency(sorted_dataset[0:split_index])
    false_freq = get_frequency(sorted_dataset[split_index:])

    child_entropy = p * calc_entropy(true_freq) + (1-p) * calc_entropy(false_freq)
    return base_entropy - child_entropy
# def checkIG(data, attr, split_point):
#     # split in 2 subsets
#     subset1 = []
#     subset2 = []
#     subset1, subset2 = split(data, attr, split_point)
#     return calc_info_gain(data[:,0], subset1[:,0], subset2[:,0])


# @param dataset NxAttr array
# @param split_info = [splitAttribute, split_point]
# @return an split_point x Attr array and an (N - split_point) x Attr array
def split(dataset, split_info):
    true_data = []
    false_data = []

    for data in dataset:
        if (split_info.match(data)):
            true_data.append(data)  
        else: 
            false_data.append(data)
    return np.array(true_data), np.array(false_data)

def find_best_split(dataset):
    best_info_gain = 0
    best_split = si.SplitInfo(None, None)

    for attr in range (len(dataset[0]) - 1):
        sorted_arr = sort_by_attr_and_label(dataset, attr)

        # find split points
        prev_split_point = sorted_arr[0][attr]
        # start checking from 1st value because splitting at 0th index will return the original array
        for row in range (1, len(sorted_arr)):

            split_point = sorted_arr[row][attr]

            if (prev_split_point != split_point):
                # check info gain
                info_gain = calc_info_gain(sorted_arr, si.SplitInfo(attr, split_point))
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_split = si.SplitInfo(attr, split_point)
            prev_split_point = split_point
    return best_split

def get_major_label(probabilities):
    
    values = list(probabilities.values())
    keys = list(probabilities.keys())
    return keys[values.index(max(values))]

def majority_element(label_list):
    idx, ctr = 0, 1
    
    for i in range(1, len(label_list)):
        if (label_list[idx] == label_list[i]):
            ctr += 1
        else:
            ctr -= 1
            if ctr == 0:
                idx = i
                ctr = 1
    
    return label_list[idx]

def get_majority_label_cross_validation(predictions):
    merged_pred = []
    for lbl in range (len(predictions[0])):
        lbl_list = []
        for i in range (len(predictions)):
            lbl_list.append(predictions[i][lbl])
        merged_pred.append(majority_element(lbl_list))
    
    return merged_pred


def get_probabilities(current_frequency, initial_frequency):

    probabilities = {}
    total = sum(current_frequency.values())

    for item in current_frequency:
        probabilities[item] = current_frequency[item] / (initial_frequency[item] if bool(initial_frequency) else total)

    return probabilities

def merge_freq(freq1, freq2):
    freq_ret = {}
    for key in freq1:
        if key in freq_ret:
            freq_ret[key] = freq_ret[key] + freq1[key]
        else:
            freq_ret[key] = freq1[key]
    for key in freq2:
        if key in freq_ret:
            freq_ret[key] = freq_ret[key] + freq2[key]
        else:
            freq_ret[key] = freq2[key]    
    return freq_ret
