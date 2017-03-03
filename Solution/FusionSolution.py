# -*- coding: utf-8 -*-
"""

@author: raoqiyu
Created on Thu Oct 15 21:37:00 2015
"""
import sys

sys.path.append("../src")

import numpy as np

from lstm import LSTM
from data import  load_fusion_data
from optimizer import ADADELTA
from layer import linear_fusion_layer,linear_activate,lstm_layer,blstm_layer,bi_linear_activate,attention_fusion_lstm_layer
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

    # compile
    model.compile(options)

    # Training
    train_err, valid_err, test_err = model.fit(trainData, validData, testData)
    del model
    return train_err, valid_err, test_err



def linear_lstm_fusion(options, trainData, validData, testData, n_layer, n_hidden):
    # Initialize Model
    model = LSTM()

    # Build Neural Network
    n_input = 1
    n_hidden = 2
    n_output = 1

    model.add(linear_fusion_layer(n_input,n_hidden))
    model.add(blstm_layer(n_hidden,n_hidden))
    model.add(blstm_layer(n_hidden, n_hidden))
    model.add(bi_linear_activate(n_hidden,n_output))
    # Choose optimizer
    adadelta = ADADELTA()
    options["optimizer"] = adadelta

    # compile
    model.compile(options)

    # Training
    train_err, valid_err, test_err = model.fit(trainData, validData, testData)
    del model
    return train_err, valid_err, test_err


def attention_lstm_fusion(options, trainData, validData, testData, n_layer, n_hidden):
    # Initialize Model
    model = LSTM()

    # Build Neural Network
    n_input = 1
    n_hidden = 1
    n_output = 1

    model.add(attention_fusion_lstm_layer(n_input,n_hidden))
    #model.add(blstm_layer(n_hidden,n_hidden))
    model.add(bi_linear_activate(n_hidden,n_output))
    # Choose optimizer
    adadelta = ADADELTA()
    options["optimizer"] = adadelta

    # compile
    model.compile(options)

    # Training
    train_err, valid_err, test_err = model.fit(trainData, validData, testData)
    del model
    return train_err, valid_err, test_err

def run():
    # trainData, validData, testData = load_avec2015_data_generated2('data/AVEC2015','features_video_appearance',
    # 'arousal', '_0.7_150_135.pkl')

    n_dim = 1
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
        "epochs": 2000,
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

    trainData, validData, testData = load_fusion_data('data', dimensions[n_dim])

    np.random.seed(123)
    n_layer = 1
    n_hidden = 1

    options["saveto"] = 'Fusion/'+dimensions[n_dim]+'/'

    #metric = linear_fusion(options, trainData, validData, testData, n_layer, n_hidden)
    #print("linear_fusion: ", metric)

    #metric = linear_lstm_fusion(options, trainData, validData, testData, n_layer, n_hidden)
    #print("linear_lstm_fusion: ", metric)


    metric = attention_lstm_fusion(options, trainData, validData, testData, n_layer, n_hidden)
    print("linear_lstm_fusion: ", metric)

if __name__ == "__main__":
    run()
