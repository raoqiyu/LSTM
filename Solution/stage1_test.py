"""
Model test

@author: raoqiyu
Created on Thu Oct 15 21:37:00 2015
"""

import sys

sys.path.append("../src")

import numpy as np

from lstm import LSTM
from data import load_avec2015_data_generated,load_avec2015_data,load_avec2015_data_norm
from optimizer import ADADELTA
from layer import hidden_layer
from layer import blstm_layer, bi_avec_activate
from utils import *

n_fea, n_dim  = 0, 1
features = ['features_video_appearance','features_video_geometric']
dimensions = ['arousal','valence']
stage1_model_names = ['Stage1/Range/App/arousal/0.75/150/model/_ccc.pkl','Stage1/Range/App/valence/0.6/150/model/_ccc.pkl']
stage1_basicmodel_names = ['Stage1/Basic/App/arousal/4BLSTM/model/_ccc.pkl','Stage1/Basic/App/valence/4BLSTM/model/_ccc.pkl']
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

np.random.seed(123)

# Initialize Model
model = LSTM()

# Build Neural Network
n_input = 84
n_layer = 3
n_hidden = 64

model.add(blstm_layer(n_input, n_hidden))
# model.add(DropOut())
for i in range(n_layer):
    model.add(blstm_layer(n_hidden, n_hidden))
    # model.add(DropOut())

model.add(bi_avec_activate(n_hidden, 1))
# model.add(DropOut())

# Choose optimizer
adadelta = ADADELTA()
options["optimizer"] = adadelta

#
model.compile(options)
print('\nStart Testing Stage1 Model \n')

for n_dim in [0,1]:
    print("\n\n")
    print("Loading data :",features[n_fea], dimensions[n_dim]) 
    trainData, validData, testData = load_avec2015_data('data',features[n_fea], dimensions[n_dim])
    
    model_name = stage1_basicmodel_names[n_dim]
    print("Loading model :", model_name)     
    model.load(model_name)

    # Evaluate

    train_RMSE, train_CC, train_CCC = predict_Stage1(model.predict, parallelize_data, trainData,
                                                     features[n_fea], dimensions[n_dim],partitions='train',generating=False,comparation=False)
    dev_RMSE, dev_CC, dev_CCC = predict_Stage1(model.predict, parallelize_data, validData + testData,
                                               features[n_fea], dimensions[n_dim],partitions='dev',generating=False,comparation=False)
    # print(train_RMSE, train_CC, train_CCC)
    # print(dev_RMSE, dev_CC, dev_CCC)
