# -*- coding: utf-8 -*-
"""

@author: raoqiyu
Created on Thu Oct 15 21:37:00 2015
"""
import sys

sys.path.append("../src")

import numpy as np

from lstm import LSTM
from data import load_avec2015_data_generated,load_avec2015_data,load_avec2015_data2,load_avec2015_data_generated2
from optimizer import ADADELTA
from layer import hidden_layer
from layer import blstm_layer, bi_avec_activate
from utils import *


def stage1(options, trainData, validData, testData, n_layer, n_hidden):
    # Initialize Model
    model = LSTM()

    # Build Neural Network
    n_input =  102
    n_hidden = n_hidden
    n_output = 1

    # logistic regression layer
    model.add(hidden_layer(n_input, n_hidden))
    #model.add(blstm_layer(n_input, n_hidden))
    # BLSTM layer
    for i in range(n_layer):
        model.add(blstm_layer(n_hidden, n_hidden))
    # linear regression layer
    model.add(bi_avec_activate(n_hidden, 1))

    # Choose optimizer
    adadelta = ADADELTA()
    options["optimizer"] = adadelta

    # compile
    model.compile(options)

    # Training
    train_err, valid_err, test_err = model.fit(trainData, validData, testData)
    del model
    return train_err, valid_err, test_err


def stage2(options, trainData, validData, testData, n_layer):
    # Initialize Model
    model = LSTM()

    n_output = 1

    for i in range(n_layer):
        model.add(blstm_layer(1, 1))
        # model.add(DropOut())

    model.add(bi_avec_activate(1, n_output))

    # Choose optimizer
    adadelta = ADADELTA()
    options["optimizer"] = adadelta

    #
    model.compile(options)

    # Unsupervised Training

    print("Unsupervised Pretraining")
    n_samples = len(trainData)
    pratrainingData = []
    for i in range(n_samples):
        pratrainingData.append((trainData[i][1], trainData[i][1]))
    print(len(pratrainingData), len(validData), len(testData))
    
    train_err, valid_err, test_err = model.fit(pratrainingData, validData, testData)

    """
    # Supervised training
    model.load(options['saveto']+'_ccc.pkl')
    print('Supervised training')
    print(len(trainData), len(validData), len(testData))
    train_err, valid_err, test_err = model.fit(trainData, validData, testData)
    """

    del model
    return train_err, valid_err, test_err


def run():
    # trainData, validData, testData = load_avec2015_data_generated2('data/AVEC2015','features_video_appearance',
    # 'arousal', '_0.7_150_135.pkl')

    n_fea, n_dim  = 2, 0
    features = ['features_video_appearance','features_video_geometric','features_audio']
    dimensions = ['arousal','valence']
    print("Data : ",features[n_fea], dimensions[n_dim])

    ##----------------------------------------
    durations = [0.6,0.7,0.75,0.8,0.9]
    skips = [100,125,150]
    n_dura = 0
    nth_skip = 0
  

    #------------------------------------
    # trainData, data1 , data2 = load_avec2015_data('data', features[n_fea], dimensions[n_dim])
    # validData = data1 + data2
    # testData = validData
    #------------------------------------




    ##-----------------------------------------------------------------------
    # Model Options
    options = {
        "epochs":2000,
        "batch_size":15,
        "valid_batch_size":5,
        "learning_rate":1e-5,
        "patience":10,
        "L1_penalty":None,
        "L2_penalty":None,
        "shuffle":False,
        "dispFreq":1,
    }
    

    # Stage 1
    filename = '_norm_0.75_150_117.pkl'
    print("Data : ",filename)
    print("Data : ", 'load_avec2015_data_generated')
    
    trainData, validData, testData = load_avec2015_data_generated('data', features[n_fea], dimensions[n_dim], filename)
    
    np.random.seed(123)
    n_layer = 2
    n_hidden = 64
    
    # /home/admin1_417/Raoqiyu/Exp/LSTM/Solution/Stage1/Range/App/arousal/0.6
    #options["saveto"] = 'Stage1/Range/App/arousal/0.7/150/model/'
    options["saveto"] = 'test/'
    
    metric = stage1(options, trainData, validData, testData, n_layer, n_hidden)
    print("Stage 1: ", metric)

    # Stage 2
    """
    filename = '_0.95_100_36.pkl'
    print("Data : ",filename)
    print("Data : ", 'load_avec2015_data_generated2')

    #trainData, validData, testData = load_avec2015_data2('data', features[n_fea], dimensions[n_dim])
    trainData, data1 , data2 = load_avec2015_data2('data', features[n_fea], dimensions[n_dim])
    validData = data1 + data2
    testData = validData

    trainData, validData, testData = load_avec2015_data_generated2('data', features[n_fea], dimensions[n_dim], filename)
    np.random.seed(123)
    #options["saveto"] = 'Stage2/Basic/Supervised/valence/model/'
    options["saveto"] = 'Stage2/Range/App/arousal/0.95/100/model/'
    n_layer = 1

    metric = stage2(options, trainData, validData, testData, n_layer)
    print("Stage 2: ", metric)
    """

if __name__ == "__main__":
    run()
