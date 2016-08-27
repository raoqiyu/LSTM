# -*- coding: utf-8 -*-
"""
main function for using lstm


@author: raoqiyu
Created on Thu Oct 15 21:37:00 2015


"""

from lstm import LSTM
from data import load_avec2015_data_generated
from optimizer import ADADELTA
from layer import lstm_layer, avec_activate, hidden_layer
from layer import blstm_layer, bi_avec_activate
from utils import *


def train_lstm(option, trainData, validData, testData, n_layer, n_hidden):
    # Initialize Model
    model = LSTM()

    # Build Neural Network
    n_input = 84
    n_hidden = n_hidden
    n_output = 1
    # model.add(lstm_layer(n_input, n_hidden))
    # model.add(DropOut())
    model.add(hidden_layer(n_input, n_hidden))
    for i in range(n_layer):
        model.add(lstm_layer(n_hidden, n_hidden))
        # model.add(DropOut())

    model.add(avec_activate(n_hidden, 1))


    # Choose optimizer
    adadelta = ADADELTA()
    options["optimizer"] = adadelta

    #
    model.compile(options)

    # Training
    train_err, valid_err, test_err = model.fit(trainData, validData, testData)
    del model
    return train_err, valid_err, test_err


def train_blstm(options, trainData, validData, testData, n_layer, n_hidden):
    # Initialize Model
    model = LSTM()

    # Build Neural Network
    n_input = 84
    n_hidden = n_hidden
    n_output = 1
    # model.add(blstm_layer(n_input, n_hidden))
    # model.add(DropOut())
    model.add(hidden_layer(n_input, n_hidden))
    for i in range(n_layer):
        model.add(blstm_layer(n_hidden, n_hidden))
        # model.add(DropOut())

    model.add(bi_avec_activate(n_hidden, 1))


    # Choose optimizer
    adadelta = ADADELTA()
    options["optimizer"] = adadelta

    #
    model.compile(options)

    # Training
    train_err, valid_err, test_err = model.fit(trainData, validData, testData)
    del model
    return train_err, valid_err, test_err


def test(option, trainData, validData, testData, n_layer, n_hidden):
    # Initialize Model
    model = LSTM()

    # Build Neural Network
    n_input = 84
    n_hidden = 64
    # n_hidden = int(n_input*n_hidden)
    n_output = 1
    model.add(lstm_layer(84, 200))
    model.add(lstm_layer(200, 160))
    # model.add(lstm_layer())
    # for i in range(n_layer):
    #     model.add(lstm_layer(n_hidden, n_hidden))
    # model.add(DropOut())

    model.add(avec_activate(160, n_output))
    # model.add(DropOut())

    # Choose optimizer
    adadelta = ADADELTA()
    options["optimizer"] = adadelta

    #
    model.compile(options)

    # Training
    train_err, valid_err, test_err = model.fit(trainData, validData, testData)
    del model
    return train_err, valid_err, test_err


def stage2_lstm(options, trainData, validData, testData, n_layer):
    # Initialize Model
    model = LSTM()

    # Build Neural Network
    n_input = 1
    n_hidden = 1
    # n_hidden = int(n_input*n_hidden)
    n_output = 1

    # model.add(hidden_layer(1,1))
    # model.add(DropOut())
    for i in range(n_layer):
        model.add(lstm_layer(1, 1))
        # model.add(DropOut())

    model.add(avec_activate(1, n_output))
    # model.add(DropOut())
    # model.add(blstm_layer(n_output, n_output))
    # model.add(bi_avec_activate(n_output, n_output))

    # Choose optimizer
    adadelta = ADADELTA()
    options["optimizer"] = adadelta

    #
    model.compile(options)

    # Training
    print("pretraining")
    n_samples = len(trainData)
    pratrainingData = []
    for i in range(n_samples):
        pratrainingData.append((trainData[i][1], trainData[i][1]))
    print(len(trainData), len(validData), len(testData))
    train_err, valid_err, test_err = model.fit(pratrainingData, validData, testData)
    del model
    return train_err, valid_err, test_err


def stage2_blstm(options, trainData, validData, testData, n_layer):
    # Initialize Model
    model = LSTM()

    # Build Neural Network
    n_input = 1
    n_hidden = 1
    # n_hidden = int(n_input*n_hidden)
    n_output = 1

    # model.add(hidden_layer(1,1))
    # model.add(DropOut())
    for i in range(n_layer):
        model.add(blstm_layer(1, 1))
        # model.add(DropOut())

    model.add(bi_avec_activate(1, n_output))
    # model.add(DropOut())
    # model.add(blstm_layer(n_output, n_output))
    # model.add(bi_avec_activate(n_output, n_output))

    # Choose optimizer
    adadelta = ADADELTA()
    options["optimizer"] = adadelta

    #
    model.compile(options)

    # Training
    print("pretraining")
    n_samples = len(trainData)
    pratrainingData = []
    for i in range(n_samples):
        pratrainingData.append((trainData[i][1], trainData[i][1]))
    print(len(trainData), len(validData), len(testData))
    train_err, valid_err, test_err = model.fit(pratrainingData, validData, testData)

    del model
    return train_err, valid_err, test_err


# noinspection PyUnresolvedReferences
def test2(model):
    s_type = ['.pkl', '_0.7_100_207.pkl', '0.7_125_162.pkl', '_0.7_150_135.pkl', '_0.75_100_171.pkl',
              '_0.75_125_135.pkl', '_0.75_150_117.pkl']
    for s in s_type:
        trainData, _, _ = load_avec2015_data_generated('data/AVEC2015', 'features_video_appearance', 'arousal', s)
        train_rmse, train_ccc = evaluate(model.predict, parallelize_data, trainData)
        print(train_rmse, train_ccc)


if __name__ == '__main__':
    n_case = 4
    n_hidden = 64
    n_layer = 3
    errors = []

    # OutPut Redirection
    # old = sys.stdout
    # sys.stdout = open('model/training/model/training/Exp/Basic/LSTM', 'w')

    ##-----------------------------------------------------------------------
    # Load data set
    # trainData, validData1, validData2 = load_avec2015_data('data/AVEC2015','features_video_appearance', 'arousal')
    # validData = validData + testData
    # testData = validData
    # trainData, validData1, validData2 = load_avec2015_data2('data/AVEC2015','features_video_appearance', 'arousal', )
    # validData = validData1 + validData2
    # testData = validData
    trainData, validData, testData = load_avec2015_data_generated('../Solution/data', 'features_video_appearance',
                                                                  'arousal', '_0.7_100_207.pkl')
    # trainData, validData, testData = load_avec2015_data_generated2('../Solution/data','features_video_appearance',
    # 'arousal', '_0.7_150_135.pkl')
    # n_sample = len(trainData)
    ##-----------------------------------------------------------------------
    # Model Options
    options = {"epochs":2000, "batch_size":15, "valid_batch_size":5, "learning_rate":1e-5, "patience":10,
               "L1_penalty":None, "L2_penalty":None, "shuffle":False, "dispFreq":1, "saveto":''.join(
            ['model/training/Exp/My/Stage2/Gen/Pretraining/Gen/2/BLSTM', '/', str(n_layer), '_layer_LSTM'])}
    print("Net ", n_layer)
    metric = train_blstm(options, trainData, validData, testData, n_layer, n_hidden)
    # print('Test')
    # sys.stdout = open('model/training/test/trainlog', 'w')
    # np.random.seed(123)
    # options["saveto"] = ''.join(['model/training/model/training/Exp/Basic/LSTM', '/', str(n_layer), '_layer_LSTM'])
    # metric = test(options, trainData, validData, testData, 2, n_hidden)
    # metric = stage2(options, trainData, validData+testData, validData+testData, n_layer, n_hidden)
##-----------------------------------------------------------------------
# Training/home/admin1_417/Raoqiyu/AVEC2015/LSTM/model/training/Exp/My/Stage2
# print("Training LSTM")
# for n_layer in range(1,5):
#     np.random.seed(123)
#     print("Net ", n_layer)
#     options["saveto"] = ''.join(['model/training/Exp/My/Stage2/Gen/Pretraining/Gen/2/LSTM', '/', str(n_layer),
# '_layer_LSTM'])
#     # metric = train_lstm(options, trainData, validData, testData, n_layer, n_hidden)
#     metric = stage2_lstm(options, trainData, validData, testData, n_layer)
#     errors.append(metric)
#
# ##-----------------------------------------------------------------------
#     print("Training BLSTM")
#     for n_layer in range(1,5):
#         np.random.seed(123)
#         options["saveto"] = ''.join(['model/training/Exp/My/Stage2/Gen/Pretraining/Gen/2/BLSTM', '/', str(n_layer),
#  '_layer_LSTM'])
#         print("Net ", n_layer)
#         metric = train_blstm(options, trainData, validData, testData, n_layer, n_hidden)
#         # metric = stage2_blstm(options, trainData, validData, testData, n_layer)
#         errors.append(metric)
#
# ##-----------------------------------------------------------------------
#     for e in errors:
#         print(e)
