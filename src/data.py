# -*- coding: utf-8 -*-
"""
@author: raoqiyu
Created on Sat Oct 17 10:34:24 2015

"""

import pickle

import numpy as np


def Load_Data(test_size=0):
    # -----------------------------Loading Data------------------------------
    print('-' * 20, 'Loading Data', '-' * 20)
    trainData, validData, testData = load_data(n_words=10000,
                                               valid_portion=0.05,
                                               maxlen=100)
    if test_size > 0:
        idx = np.arange(len(testData[0]))
        np.random.shuffle(idx)
        idx = idx[:test_size]
        testData = ([testData[0][n] for n in idx],
                    [testData[1][n] for n in idx])

    # number of classes

    print('%d train examples' % len(trainData[0]))
    print('%d valid examples' % len(validData[0]))
    print('%d test  examples' % len(testData[0]))
    return trainData, validData, testData

def load_fusion_data(path, label,data_base):
    # -----------------------------Loading Data------------------------------
    if data_base != 'AVEC2015':
        set_path = path + '/recola_data/training/fusion/' + label +  '/'
        trainfile = set_path + 'train' + label.capitalize()+'.pkl'
        validfile = set_path + 'valid' + label.capitalize()+'.pkl'
        testfile  = set_path + 'test' + label.capitalize()+'.pkl'
    else:
        set_path = path + '/avec_data/training/fusion/' + label + '/'
        trainfile = set_path + 'train' + label.capitalize() + '.pkl'
        validfile = set_path + 'dev' + label.capitalize() + '.pkl'
        testfile = set_path + 'dev' + label.capitalize() + '.pkl'
    print('-' * 20, 'Loading fusion Data', '-' * 20)
    print(set_path)
    print(trainfile)
    print(validfile)
    print(testfile)

    f_train = open(trainfile, 'rb')
    trainData = pickle.load(f_train)
    f_train.close()

    f_val = open(validfile, 'rb')
    validData = pickle.load(f_val)
    f_val.close()

    f_test = open(testfile, 'rb')
    testData = pickle.load(f_test)
    f_test.close()

    print('%d train examples' % len(trainData),len(trainData[0]), len(trainData[0][0]),len(trainData[0][0][0]))
    print('%d valid examples' % len(validData),len(validData[0]), len(validData[0][0]),len(validData[0][0][0]))
    print('%d test  examples' % len(testData),len(testData[0]), len(testData[0][0]),len(testData[0][0][0]))
    return trainData, validData, testData


def load_recola_data(path, feature, label):
    # -----------------------------Loading Data------------------------------
    print('-' * 20, 'Loading recola Data', '-' * 20)

    filepath = '/'.join([path, 'recola_data/training', feature, label])

    f_train = open(filepath + '/' + 'train' + label.capitalize() + '.pkl', 'rb')
    trainData = pickle.load(f_train)
    f_train.close()

    f_val = open(filepath + '/' + 'valid' + label.capitalize() + '.pkl', 'rb')
    validData = pickle.load(f_val)
    f_val.close()

    f_test = open(filepath + '/' + 'test' + label.capitalize() + '.pkl', 'rb')
    testData = pickle.load(f_test)
    f_test.close()

    print('%d train examples' % len(trainData))
    print('%d valid examples' % len(validData))
    print('%d test  examples' % len(testData))
    return trainData, validData, testData

def load_recola_data2(path, feature, label):
    # -----------------------------Loading Data------------------------------
    print('-' * 20, 'Loading recola Data', '-' * 20)

    filepath = '/'.join([path, 'recola_data/training', feature, label])

    f_train = open(filepath + '/' + 'train' + label.capitalize() + 'Second.pkl', 'rb')
    trainData = pickle.load(f_train)
    f_train.close()

    f_val = open(filepath + '/' + 'valid' + label.capitalize() + 'Second.pkl', 'rb')
    validData = pickle.load(f_val)
    f_val.close()

    f_test = open(filepath + '/' + 'test' + label.capitalize() + 'Second.pkl', 'rb')
    testData = pickle.load(f_test)
    f_test.close()

    print('%d train examples' % len(trainData))
    print('%d valid examples' % len(validData))
    print('%d test  examples' % len(testData))
    return trainData, validData, testData

def load_recola_data_generated2(path, feature, label, s_type):
    # -----------------------------Loading Data------------------------------
    print('-' * 20, 'Loading recola Data', '-' * 20)
    print('generated_Stage2' + s_type)

    filepath = '/'.join([path, 'recola_data/training', feature, label])

    f_val = open(filepath + '/' + 'valid' + label.capitalize() + 'Second.pkl', 'rb')
    validData = pickle.load(f_val)
    f_val.close()

    f_test = open(filepath + '/' + 'test' + label.capitalize() + 'Second.pkl', 'rb')
    testData = pickle.load(f_test)
    f_test.close()

    f_generated = open(filepath + '/' + 'generated_train_Stage2' + s_type, 'rb')
    generatedData = pickle.load(f_generated)
    f_generated.close()

    print('%d generated train examples' % len(generatedData), ":", len(generatedData), len(generatedData[0][0]))
    print('%d original valid examples' % len(validData), ":", len(validData), len(validData[0][0]))
    print('%d test  examples' % len(testData), ":", len(testData), len(testData[0][0]))
    return generatedData, validData, testData

def load_recola_data_generated(path, feature, label, s_type):
    # -----------------------------Loading Data------------------------------
    print('-' * 20, 'Loading recola Data', '-' * 20)
    print('generated_train' + s_type)

    filepath = '/'.join([path, 'recola_data/training', feature, label])

    f_val = open(filepath + '/' + 'valid' + label.capitalize() + '.pkl', 'rb')
    validData = pickle.load(f_val)
    f_val.close()

    f_test = open(filepath + '/' + 'test' + label.capitalize() + '.pkl', 'rb')
    testData = pickle.load(f_test)
    f_test.close()

    f_generated = open(filepath + '/' + 'generated_train' + s_type, 'rb')
    generatedData = pickle.load(f_generated)
    f_generated.close()

    print('%d generated train examples' % len(generatedData), ":", len(generatedData), len(generatedData[0][0]))
    print('%d original train examples' % len(validData), ":", len(validData), len(validData[0][0]))
    print('%d valid  examples' % len(testData), ":", len(testData), len(testData[0][0]))
    return generatedData, validData, testData


def load_avec2015_data(path, feature, label, valid_size=3):
    # -----------------------------Loading Data------------------------------
    print('-' * 20, 'Loading Data', '-' * 20)

    filepath = '/'.join([path, 'avec2015_data/training', feature, label])

    f_train = open(filepath + '/' + 'train' + label.capitalize() + '.pkl', 'rb')
    trainData = pickle.load(f_train)
    f_train.close()

    f_dev = open(filepath + '/' + 'dev' + label.capitalize() + '.pkl', 'rb')
    validData = pickle.load(f_dev)
    f_dev.close()
    testData, validData = validData[valid_size:], validData[:valid_size]
    print('%d train examples' % len(trainData))
    print('%d valid examples' % len(validData))
    print('%d test  examples' % len(testData))
    return trainData, validData, testData


def load_avec2015_data2(path, feature, label, valid_size=3):
    # -----------------------------Loading Data------------------------------
    print('-' * 20, 'Loading Data', '-' * 20)

    filepath = '/'.join([path, 'avec2015_data/training', feature, label])

    f_train = open(filepath + '/' + 'train' + label.capitalize() + 'Second.pkl', 'rb')
    trainData = pickle.load(f_train)
    f_train.close()

    f_dev = open(filepath + '/' + 'dev' + label.capitalize() + 'Second.pkl', 'rb')
    validData = pickle.load(f_dev)
    f_dev.close()
    testData, validData = validData[valid_size:], validData[:valid_size]
    print('%d train examples' % len(trainData))
    print('%d valid examples' % len(validData))
    print('%d test  examples' % len(testData))
    return trainData, validData, testData
    
def load_avec2015_data_norm(path, feature, label, valid_size=3):
    # -----------------------------Loading Data------------------------------
    print('-' * 20, 'Loading Data', '-' * 20)

    filepath = '/'.join([path, 'avec2015_data/training', feature, label])
    print("trainData: _norm.pkl")
    f_train = open(filepath + '/' + 'train' + label.capitalize() + '_normed.pkl', 'rb')
    trainData = pickle.load(f_train)
    f_train.close()

    print("devData: _norm.pkl")
    f_dev = open(filepath + '/' + 'dev' + label.capitalize() + '_normed.pkl', 'rb')
    validData = pickle.load(f_dev)
    f_dev.close()
    testData, validData = validData[valid_size:], validData[:valid_size]
    print('%d train examples' % len(trainData))
    print('%d valid examples' % len(validData))
    print('%d test  examples' % len(testData))
    return trainData, validData, testData


def load_avec2015_data_pooling(path, feature, label, valid_size=3):
    # -----------------------------Loading Data------------------------------
    print('-' * 20, 'Loading Data', '-' * 20)

    filepath = '/'.join([path, 'avec2015_data/training', feature, label])
    print("trainData: _pooling.pkl")
    f_train = open(filepath + '/' + 'train' + label.capitalize() + '_pooling.pkl', 'rb')
    trainData = pickle.load(f_train)
    f_train.close()

    print("devData: .pkl")
    f_dev = open(filepath + '/' + 'dev' + label.capitalize() + '.pkl', 'rb')
    validData = pickle.load(f_dev)
    f_dev.close()
    testData, validData = validData[valid_size:], validData[:valid_size]
    print('%d train examples' % len(trainData))
    print('%d valid examples' % len(validData))
    print('%d test  examples' % len(testData))
    return trainData, validData, testData


def load_avec2015_data_generated(path, feature, label, s_type):
    # -----------------------------Loading Data------------------------------
    print('-' * 20, 'Loading Data', '-' * 20)
    print('generated_train' + s_type)
    filepath = '/'.join([path, 'avec2015_data/training', feature, label])

    f_train = open(filepath + '/' + 'train' + label.capitalize() + '.pkl', 'rb')
    trainData = pickle.load(f_train)
    f_train.close()

    f_dev = open(filepath + '/' + 'dev' + label.capitalize() + '.pkl', 'rb')
    validData = pickle.load(f_dev)
    f_dev.close()

    f_generated = open(filepath + '/' + 'generated_train' + s_type, 'rb')
    generatedData = pickle.load(f_generated)
    f_generated.close()

    print('%d generated train examples' % len(generatedData), ":", len(generatedData), len(generatedData[0][0]))
    print('%d original train examples' % len(trainData), ":", len(trainData), len(trainData[0][0]))
    print('%d valid  examples' % len(validData), ":", len(validData), len(validData[0][0]))
    return generatedData, trainData, validData


def load_avec2015_data_generated2(path, feature, label, s_type):
    # -----------------------------Loading Data------------------------------
    print('-' * 20, 'avec2015_data/Loading Data', '-' * 20)
    print('generated_Stage2' + s_type)
    filepath = '/'.join([path, 'training', feature, label])

    f_train = open(filepath + '/' + 'train' + label.capitalize() + 'Second.pkl', 'rb')
    trainData = pickle.load(f_train)
    f_train.close()

    f_dev = open(filepath + '/' + 'dev' + label.capitalize() + 'Second.pkl', 'rb')
    validData = pickle.load(f_dev)
    f_dev.close()

    f_generated = open(filepath + '/' + 'generated_train_Stage2' + s_type, 'rb')
    generatedData = pickle.load(f_generated)
    f_generated.close()

    print('%d generated train examples' % len(generatedData), ":", len(generatedData), len(generatedData[0][0]))
    print('%d original train examples' % len(trainData), ":", len(trainData), len(trainData[0][0]))
    print('%d valid  examples' % len(validData), ":", len(validData), len(validData[0][0]))
    return generatedData, trainData, validData


def GaussianNorm(data, filename):
    """
    DO gaussian normalization
    :param data:
    :return:
    """
    print("Doing Gaussian Normalization")
    n_samples = len(data)
    n_steps = len(data[0][0])
    n_features = len(data[0][0][0])
    # mean and var
    params = []
    # Loop for each data
    for sample in range(n_samples):
        p = []
        for n_fea in range(n_features):
            d = np.array(data[sample][0])
            d_mean = np.mean(d[:, n_fea])
            d_var = np.mean((d[:, n_fea] - d_mean) ** 2)
            d_var_square = d_var ** 2
            # Gaussian basis functions
            for n_s in range(n_steps):
                x = data[sample][0][n_s][n_fea]
                x = (x - d_mean) ** 2 / 2 / d_var_square
                x = np.exp(-x)
                data[sample][0][n_s][n_fea] = x
            p.append((d_mean, d_var))
        params.append(p)

    with open("data/AVEC2015/meanVar.pkl", 'wb') as f:
        pickle.dump(params, f)

    with open("data/AVEC2015/training/features_video_appearance/arousal/" + filename, 'wb') as f:
        pickle.dump(data, f)

def Normalizs(data, filename):
    """
    DO normalization
    :param data:
    :return:
    """
    print("Doing Normalization")
    n_samples = len(data)
    n_steps = len(data[0][0])
    n_features = len(data[0][0][0])
    # mean and var
    params = []
    # Loop for each data
    for sample in range(n_samples):
            # basis functions
        for n_s in range(n_steps):
            d = np.array(data[sample][0][n_s])
            d -= d.min()
            d = d/(d.max()-d.min())
            data[sample][0][n_s] = d.tolist()
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def Z_ScoreNorm(data, filename):
    """
    DO gaussian normalization
    :param data:
    :return:
    """
    print("Doing Z_Score Normalization")
    n_samples = len(data)
    n_steps = len(data[0][0])
    n_features = len(data[0][0][0])
    # mean and var
    params = []
    # Loop for each data
    for sample in range(n_samples):
        p = []
        for n_fea in range(n_features):
            d = np.array(data[sample][0])
            d_mean = np.mean(d[:, n_fea])
            d_var = np.mean((d[:, n_fea] - d_mean) ** 2)
            d_var_square = d_var ** 2
            # Gaussian basis functions
            for n_s in range(n_steps):
                x = data[sample][0][n_s][n_fea]
                x = (x - d_mean)/ d_var_square
                data[sample][0][n_s][n_fea] = x
            p.append((d_mean, d_var))
        params.append(p)

    with open(filename, 'wb') as f:
        pickle.dump(data, f)