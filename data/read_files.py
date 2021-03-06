from collections import OrderedDict
import sys
if sys.version_info[0]==2:
    import cPickle as pickle
else:
    import pickle
import pickle
import h5py
import os
import json
import shutil
import numpy as np
import csv

def read_from_tsv(filename):
    with open('data/'+filename, 'r') as mycsvfile:
        files = csv.reader(mycsvfile,delimiter='\t')
        dataset = list()
        for row in files:
            dataset.append(row)
    return dataset


def save_in_pickle(file,array):
    with open(file+'.txt', 'wb') as handle:
        pickle.dump(array, handle)

def read_from_pickle(file):
    with open(file+'.txt', 'rb') as handle:
        if sys.version_info[0] == 2:
            b = pickle.loads(handle.read())
        else:
            b = pickle.loads(handle.read(),encoding='latin1')
    return b


def save_in_json(filename, array):
    with open(filename+'.txt', 'w') as outfile:
        json.dump(array, outfile)
    outfile.close()

def read_from_json(filename):
    with open(filename+'.txt', 'r') as outfile:
        data = json.load(outfile)
    outfile.close()
    return data

def read_from_dir(path):
    data =open(path).read()
    return data

def textfile2list(path):
    data = read_from_dir(path)
    list_new =list()
    for line in data.splitlines():
        list_new.append(line)
    return list_new

def movefiles(old_address,new_address,dir_simples,abbr):
    i = 0
    for dir_simple in dir_simples:
        desti = new_address+dir_simple +abbr
        shutil.copy(old_address+dir_simple+abbr,desti)
        i+=1

def load_h5py_data(filename):
    data = list()
    with h5py.File( filename + '.hdf5', 'r') as hf:
        print("List of arrays in this file: \n", hf.keys())
        for key in hf.keys():
            x = hf.get(key)
            x_data = np.array(x)
            print x_data.shape
            del x
            data.append(x_data)

    return data

def save_h5py_data(filename,input_names,inputs,data_format):
    f = h5py.File(filename+".hdf5", "w")
    for inut_index in range(len(input)):
        dset_char = f.create_dataset(input_names[inut_index], data=inputs[inut_index], dtype=data_format[inut_index])
        
def index2vector(y, nb_classes=None):
    y = np.array(y, dtype='int').ravel()
    if not nb_classes:
        nb_classes = np.max(y) + 1
    n = y.shape[0]
    y1 = np.copy(y)
    y = [0 if x==-1 else x for x in y1]

    categorical = np.eye(nb_classes)[y]
    for item in range(n):
        if y1[item]==-1:
            categorical[item]  = np.zeros(nb_classes)
    return categorical

def counterList2Dict (counter_list):
    dict_new = dict()
    for item in counter_list:
        dict_new[item[0]]=item[1]
    return dict_new

