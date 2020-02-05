import numpy as np
from helpers import *

def analyse(filename):
  attr, label = read_file(filename)
  print(filename)
  # type of attributes
  print("Type of attribute is ", attr.dtype)

  # range of each attribute
  attr_max = np.amax(attr, axis = 0)
  attr_min = np.amin(attr, axis = 0)
  for i in range(len(attr[0])):
    print("Range of attribute ", i, " is {} - {}".format(attr_min[i], attr_max[i]))

  # range of labels
  dataset = get_data(attr, label)
  class_frequency = get_frequency(dataset)
  for key in class_frequency:
    print(chr(key), ": ", class_frequency[key])

# Difference between train_full and train_sub
def class_distribution(file1, file2):
  attr1, label1 = read_file(file1)
  attr2, label2 = read_file(file2)
  dataset1 = get_data(attr1, label1)
  dataset2 = get_data(attr2, label2)

  frequency1 = get_frequency(dataset1)
  frequency2 = get_frequency(dataset2)

  distr1 = {}
  distr2 = {}

  for key in frequency1:
    distr1[chr(key)] = frequency1[key]/len(dataset1)
  for key in frequency2:
    distr2[chr(key)] = frequency2[key]/len(dataset2)

  print("Distribution in ", file1, ": ")
  for key in distr1:
    print(key, ": ", round(distr1[key]*100, 2), "%")
  print("Distribution in ", file2, ": ")
  for key in distr2:
    print(key, ": ", round(distr2[key]*100, 2), "%")


# Difference of labels between train_full and train_noisy
def proportion_of_difference(file1, file2):
  attr1, label1 = read_file(file1)
  attr2, label2 = read_file(file2)
  dataset1 = get_data(attr1, label1)
  dataset2 = get_data(attr2, label2)

  frequency1 = get_frequency(dataset1)
  frequency2 = get_frequency(dataset2)
  difference = {}
  
  for key in frequency2:
    diff = frequency2[key] - frequency1[key]
    difference[chr(key)] = diff if diff >= 0 else -(diff)
  
  total_diff = 0
  for key in difference:
    print("Difference of label ", key, ": ", difference[key])
    total_diff += difference[key] 
  
  proportion = total_diff / len(dataset2)
  print("Proportion of difference is: ", round(proportion*100, 2), "%")

def map_att_to_label(filename):
  attr, label = read_file(filename)
  att_label = {}
  for i in range(len(attr)):
    uniq_att = ",".join([str(att) for att in attr[i]])
    if uniq_att not in att_label:
      att_label[uniq_att] = {}
    if label[i] not in att_label[uniq_att]:
      att_label[uniq_att][label[i]] = 0
    att_label[uniq_att][label[i]] += 1
  
  inconsistent_count = 0
  for key in att_label:
    # print(key, "; ", att_label[key])
    if len(att_label[key].keys()) > 1:
      inconsistent_count += 1
      # print("Inconsistent class: ", key, ": ", att_label[key])
    # else:
    #   for label in att_label[key]:
    #     if att_label[key][label] > 1:
    #       print(key, ": ", att_label[key])
  print("Number of attribute strings that have more than one class labels: ", inconsistent_count)
  return att_label

def match_att_label(file1, file2):
  att_label1 = map_att_to_label(file1)
  att_label2 = map_att_to_label(file2)

  match_count = 0
  mismatch_count = 0
  for att in att_label1:
    if att_label1[att].keys() != att_label2[att].keys():
      mismatch_count += 1
      # print(att, att_label1[att], att_label2[att])
    else:
      match_count += 1
  print("Number of instances with matching labels: ", match_count)
  print("Number of instances with mismatched labels: ", mismatch_count)


# analyse("data/train_full.txt")
# analyse("data/train_noisy.txt")
# analyse("data/train_sub.txt")

# class_distribution("data/train_full.txt", "data/train_sub.txt")
# proportion_of_difference("data/train_full.txt", "data/train_noisy.txt")

# map_att_to_label("data/train_noisy.txt")
# match_att_label("data/train_full.txt", "data/train_noisy.txt")