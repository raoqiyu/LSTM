# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 22:22:07 2015

@author: wahah
"""
import sys,os
sys.path.append("../../src")

import pickle

from data import load_avec2015_data,load_avec2015_data2,Normalizs,Z_ScoreNorm,load_avec2015_data_norm,load_recola_data
from utils import generateData

#trainData, validData, testData = load_recola_data('./',
#                                                     'features_video', 'valence')
trainData, validData, testData = load_avec2015_data('./',
                                                     'features_audio', 'valence')

#Z_ScoreNorm(trainData, "training/features_audio/arousal/trainArousal_norm.pkl")
#Z_ScoreNorm(validData+testData, "training/features_audio/arousal/devArousal_norm.pkl")
# print("Done!")
#Normalizs(trainData, "trainArousal_normed.pkl")
#Normalizs(validData+testData, "devArousal_normed.pkl")

#--------------------------------------------------------

#durations = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
#n_skips = [100,125,150]
durations=[0.6]
n_skips=[150]
for duration in durations:
   for n_skip in n_skips:
       print(duration, n_skip)
       shuffled_trainData = generateData(trainData, duration, n_skip)
       n_sample = len(shuffled_trainData)

       filename =  "avec_data/training/features_audio/valence/generated_train_" + str(duration) + "_" + str(n_skip) + "_" + str(n_sample) + ".pkl"
       if os.path.exists(filename):
            print("File exits. continue...")
            continue
       with open(filename, 'wb') as f:
           pickle.dump(shuffled_trainData, f)

#----------------------------------------------------------------------
# shuffled_trainData = generateData(validData+testData, duration, n_skip)
# n_sample = len(shuffled_trainData)
# print(n_sample)
# with open("data/AVEC2015/training/features_video_appearance/arousal/generated_dev_"+str(duration)+"_"+str(
# n_skip)+"_"+str(n_sample)+"_stage2.pkl", 'wb') as f:
#     pickle.dump(shuffled_trainData, f)



#----------------------------------------------------------------------
    # X = parallelize_data(x)

    # trainData = temporal_pooling(trainData, window_size=1.,with_label=False)
    # with open("data/AVEC2015/training/features_video_appearance/arousal/trainArousal_pooling_without_label.pkl",
    # 'wb') as f:
    #     pickle.dump(trainData, f)
    #
    # devData = temporal_pooling(validData+testData, window_size=1.,with_label=False)
    # with open("data/AVEC2015/training/features_video_appearance/arousal/devArousal_pooling_without_label.pkl",
    # 'wb') as f:
    #     pickle.dump(devData, f)


    # new_trainData = []
    # for i in range(len(trainData)):
    #     new_x = add_noise(trainData[i][0])
    #     new_y = trainData[i][1]
    #     new_trainData.append( (new_x, new_y) )
    # with open("data/AVEC2015/training/features_video_appearance/arousal/trainArousal_noise.pkl", 'wb') as f:
    #     pickle.dump(new_trainData, f)

    # trainData1, _, _ = load_avec2015_data_generated('data/AVEC2015','features_video_appearance', 'arousal', ".pkl")
    #
    # trainData2, _, _ = load_avec2015_data_generated('data/AVEC2015','features_video_appearance', 'arousal',
    # '_0.7_150_135.pkl')
    #
    # trainData3, _, _ = load_avec2015_data_generated('data/AVEC2015','features_video_appearance', 'arousal',
    # '_0.75_125_135.pkl')
    #
    #
    # # load_avec2015_data_generated('data/AVEC2015','features_video_appearance', 'arousal',
    # '_0.75_150_117.pkl')load_avec2015_data_generated('data/AVEC2015','features_video_appearance', 'arousal', ".pkl")
    # load_avec2015_data_generated('data/AVEC2015','features_video_appearance', 'arousal', '_0.7_100_207.pkl')
    # load_avec2015_data_generated('data/AVEC2015','features_video_appearance', 'arousal', '_0.7_125_162.pkl')
    # load_avec2015_data_generated('data/AVEC2015','features_video_appearance', 'arousal', '_0.7_150_135.pkl')
    # load_avec2015_data_generated('data/AVEC2015','features_video_appearance', 'arousal', '_0.75_100_171.pkl')
    # load_avec2015_data_generated('data/AVEC2015','features_video_appearance', 'arousal', '_0.75_125_135.pkl')
    # load_avec2015_data_generated('data/AVEC2015','features_video_appearance', 'arousal', '_0.75_150_117.pkl')
