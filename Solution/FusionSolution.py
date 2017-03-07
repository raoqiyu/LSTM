# -*- coding: utf-8 -*-
"""

@author: raoqiyu
Created on Thu Oct 15 21:37:00 2015
"""
import sys
from sklearn.svm import LinearSVR
import pickle

sys.path.append("../src")

import numpy as np

from lstm import LSTM
from data import  load_fusion_data
from optimizer import ADADELTA
from layer import linear_fusion_layer,linear_activate,lstm_layer,blstm_layer,bi_linear_activate,attention_fusion_lstm_layer
from layer import attention_fusion_blstm_layer,attention_fusion_blstm_layer2
from utils import *


def linear_fusion(options, trainData, validData, testData, n_layer, n_hidden):
    # Initialize Model
    model = LSTM()

    # Build Neural Network
    n_input = 1
    n_hidden = n_hidden
    n_output = 1

    model.add(linear_fusion_layer(n_input,n_output))

    # Choose optimizer
    adadelta = ADADELTA()
    options["optimizer"] = adadelta
    options["saveto"] += 'LinearRegression/model/'

    # compile
    model.compile(options,Fusion=True)

    # Training
    train_err, valid_err, test_err = model.fit(trainData, validData, testData)
    del model
    return train_err, valid_err, test_err

def linear_SVR(dim,data_base):
    if data_base != 'AVEC2015':
        set_path = 'data/recola_data/training/fusion2/' + dim +  '/'
        trainfile = set_path + 'train' + dim.capitalize()+'linear.pkl'
        validfile = set_path + 'valid' + dim.capitalize()+'linear.pkl'
        testfile  = set_path + 'test' + dim.capitalize()+'linear.pkl'
    else:
        set_path = 'data/avec2015_data/training/fusion2/' + dim + '/'
        trainfile = set_path + 'train' + dim.capitalize() + 'linear.pkl'
        validfile = set_path + 'dev' + dim.capitalize() + 'linear.pkl'
        testfile = set_path + 'dev' + dim.capitalize() + 'linear.pkl'
    fp = open(trainfile,'rb')
    trainX, trainY = pickle.load(fp), pickle.load(fp)

    fp.close()
    fp = open(validfile, 'rb')
    validX, validY = pickle.load(fp), pickle.load(fp)

    fp.close()
    fp = open(testfile, 'rb')
    testX, testY = pickle.load(fp), pickle.load(fp)
    fp.close()
    best_c, best_CCC = -1, -1
    for c in np.arange(0.01,10, 0.2):
        clf = LinearSVR(C=c, max_iter=500)
        clf.fit(trainX,trainY)
        train_pred = clf.predict(trainX)
        valid_pred = clf.predict(validX)
        test_pred  = clf.predict(testX)
        trainRMSE, trainCC, trainCCC = rater_statistics(train_pred, np.array(trainY))
        validRMSE, validCC, validCCC = rater_statistics(valid_pred, np.array(validY))
        testRMSE, testCC, testCCC = rater_statistics(test_pred, np.array(testY))
        if validCCC > best_CCC:
            best_c  = c
            best_CCC = validCCC
        print("Train Data:",trainRMSE, trainCC, trainCCC)
        print("Valid Data:",validRMSE, validCC, validCCC)
        print("Test  Data:",testRMSE, testCC, testCCC)
        print()
    clf = LinearSVR(C=best_c,max_iter=500)
    clf.fit(trainX, trainY)
    train_pred = clf.predict(trainX)
    valid_pred = clf.predict(validX)
    test_pred = clf.predict(testX)
    trainRMSE, trainCC, trainCCC = rater_statistics(train_pred, np.array(trainY))
    validRMSE, validCC, validCCC = rater_statistics(valid_pred, np.array(validY))
    testRMSE, testCC, testCCC = rater_statistics(test_pred, np.array(testY))

    print('The best c is:', best_c)
    print("Train Data:", trainRMSE, trainCC, trainCCC)
    print("Valid Data:", validRMSE, validCC, validCCC)
    print("Test  Data:", testRMSE, testCC, testCCC)


def linear_blstm_fusion(options, trainData, validData, testData, n_layer, n_hidden):
    # Initialize Model
    model = LSTM()

    # Build Neural Network
    n_input = 1
    #n_hidden = 2
    n_output = 1

    model.add(linear_fusion_layer(n_input,1))
    model.add(blstm_layer(1,n_hidden))
    #model.add(blstm_layer(n_hidden, n_hidden))
    model.add(bi_linear_activate(n_hidden,n_output))
    # Choose optimizer
    adadelta = ADADELTA()
    options["optimizer"] = adadelta
    options["saveto"] += 'LinearBLSTM/' + str(n_hidden) + 'node/model/'

    # compile
    model.compile(options,Fusion=True)

    # Training
    train_err, valid_err, test_err = model.fit(trainData, validData, testData)
    del model
    return train_err, valid_err, test_err


def attention_blstm_fusion(options, trainData, validData, testData, n_layer, n_hidden):
    # Initialize Model
    model = LSTM()

    # Build Neural Network
    n_input = 1
    #n_hidden = 1
    n_output = 1

    #model.add(attention_fusion_lstm_layer(n_input,n_hidden))
    model.add(attention_fusion_blstm_layer2(n_input,n_hidden))
    #for i in range(n_layer):
    #    model.add(blstm_layer(n_hidden,n_hidden))
    #model.add(blstm_layer(n_hidden, n_hidden))
    #model.add(blstm_layer(n_hidden, n_hidden))
    model.add(bi_linear_activate(n_hidden,n_output))
    # Choose optimizer
    adadelta = ADADELTA()
    options["optimizer"] = adadelta
    options["saveto"] += 'AttentionBLSTM/'+str(n_hidden)+'node/model/'

    # compile
    model.compile(options,Fusion=True)

    # Training
    train_err, valid_err, test_err = model.fit(trainData, validData, testData)
    del model
    return train_err, valid_err, test_err

def run():
    # trainData, validData, testData = load_avec2015_data_generated2('data/AVEC2015','features_video_appearance',
    # 'arousal', '_0.7_150_135.pkl')
    data_base = 'AVEC2015'
    n_dim = 0
    print(data_base)
    dimensions = ['arousal', 'valence']
    print("Fusion Data Emotional Dimension: ", dimensions[n_dim])

    ##----------------------------------------
    durations = [0.6, 0.7, 0.75, 0.8, 0.9]
    skips = [100, 125, 150]
    n_dura = 0
    nth_skip = 0

    ##-----------------------------------------------------------------------
    # Model Options
    options = {
        "epochs": 500,
        "batch_size": 3,
        "valid_batch_size": 5,
        "learning_rate": 1e-5,
        "patience": 10,
        "L1_penalty": None,
        "L2_penalty": None,
        "shuffle": False,
        "dispFreq": 1,
    }

    # Stage 1
    print("Data : ", dimensions[n_dim])
    print("Data : ", 'load_fusion_data')

    trainData, validData, testData = load_fusion_data('data', dimensions[n_dim],data_base)

    np.random.seed(123)
    n_layer = 1
    n_hidden = 8

    options["saveto"] = 'Fusion/'+data_base + '/' + dimensions[n_dim]+'/'

    #linear_SVR(dimensions[n_dim],data_base)
    #metric = linear_fusion(options, trainData, validData, testData, n_layer, n_hidden)
    #print("linear_fusion: ", metric)

    metric = linear_blstm_fusion(options, trainData, validData, testData, n_layer, n_hidden)
    print("linear_lstm_fusion: ", metric)


    #metric = attention_blstm_fusion(options, trainData, validData, testData, n_layer, n_hidden)
    #print("attention_blstm_fusion: ", metric)

if __name__ == "__main__":
    run()
