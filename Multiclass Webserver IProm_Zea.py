# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 16:01:47 2021

@author: Shujaat
"""
import os
import sys

from focal_loss import BinaryFocalLoss
import os
import sys
import argparse
import numpy as np
from tensorflow.keras.models import model_from_json
import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D,AveragePooling1D
from tensorflow.keras.optimizers import SGD
#from tensorflow.keras.layers.wrappers import Bidirectional, TimeDistributed
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, BatchNormalization
from tensorflow.keras.models import Model
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import regularizers
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from tensorflow.keras import initializers
from tensorflow.keras.layers import Activation, Dense, Add
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import learning_curve
from sklearn import metrics
from sklearn.metrics import auc
from tensorflow.keras.layers import LSTM
from focal_loss import BinaryFocalLoss
from Bio import SeqIO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import StratifiedKFold

def get_model():


    input_shape = (251,4)
    inputs = Input(shape = input_shape)

    convLayer = Conv1D(filters = 32, kernel_size = 5,activation = 'relu',input_shape = input_shape, kernel_regularizer = regularizers.l2(1e-5))(inputs)#, bias_regularizer = regularizers.l2(1e-4))(inputs)
    poolingLayer1 = MaxPooling1D(pool_size = 2, strides=2)(convLayer)

    convLayer2 = Conv1D(filters =32, kernel_size = 4,activation = 'relu',kernel_regularizer = regularizers.l2(1e-5),
                        bias_regularizer = regularizers.l2(1e-5))(poolingLayer1)#(dropoutLayer0)
    poolingLayer2 = MaxPooling1D(pool_size = 2, strides=2)(convLayer2)

    convLayer3 = Conv1D(filters =64, kernel_size = 5,activation = 'relu',kernel_regularizer = regularizers.l2(1e-4),
                      bias_regularizer = regularizers.l2(1e-4))(poolingLayer2)#(dropoutLayer0)
    poolingLayer3 = MaxPooling1D(pool_size = 2, strides=2)(convLayer3)

    flattenLayer = Flatten()(poolingLayer3)#(dropoutLayer2 )#
    dropoutLayer = Dropout(0.5)(flattenLayer)
    denseLayer2 = Dense(256, activation = 'relu',kernel_regularizer = regularizers.l2(1e-5),bias_regularizer = regularizers.l2(1e-5))(dropoutLayer )
    dropoutLayer1 = Dropout(0.5)( denseLayer2 )
    denseLayer3 = Dense(64, activation = 'relu',kernel_regularizer = regularizers.l2(1e-5),bias_regularizer = regularizers.l2(1e-5))(dropoutLayer1 )
    dropoutLayer2 = Dropout(0.5)( denseLayer3 )
    outLayer = Dense(3, activation='sigmoid')(dropoutLayer2)

    model2 = Model(inputs = inputs, outputs = outLayer)


    model2.compile(loss='categorical_crossentropy',optimizer=optimizers.Adamax(lr=0.0048,
                                                                            beta_1=0.9,
                                                                            beta_2=0.999,
                                                                            epsilon=1e-07,
                                                                            decay=0.0),metrics=['accuracy'])
    #Early Stoping

    return model2


modelProMN = get_model()
modelProMN.load_weights('D:/Research papers/iPromZea/Multiclass model weights.h5')

import numpy as np
def encode_seq(s):
    Encode = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1]}
    return np.array([Encode[x] for x in s])


X1 = {}
accumulator=0

Strng2=["AGATTTAATGTTTTACTGTTGGAACGAGATGTTTTGTTAGTGCCCTAAATTATATAAAATATATATTTATTTTTAAATTATAGAATATTTTTATAGGTCACTTCCAACCTGGCGTTATGCAGAAGCACAAATCTGATCCAAAATCCAAACAGTCTTCCCGATCAAAACACCTATGGGCTTGGACTTGTTTGTACACATATAAATCTAACTTAATCCATATAGAGAAAGTTAAACTGAAATTTATAATTTAA",
       "GAACAGGAGGGAGCATGGAGTGCACTTCTTGTTCTAGTATATTGAGGCCTCGTTTGGTAGAGGCTCCATGATTCTCTAATACAGTGATTCTGAGTGATTTTCTATTGCAAGTGAATCTATTTGACGAAAACTGTTTGATAAATAGGCTGTGAAGTGATTTTTGAAGGATTAAAGAGTGAGAAGCAGGTTGAGAGTGGTGGGAAGCAGGTTTTTTTGCTCCCAATTTCTAGTACAAAGTAGAGACTAGATTC",
       "GGGGCGCGGCGGCCAACTGCCTTGCCCTTGCACTGATGGATGCCGGGACCCTAGTCCCGAAGACGGATGGGTTTGGGCGTTTGGCTGGGAGCAGGATGCACGGGAGCCACTCGTTCGGTCGTTCCTGCGCCGCGATGCAGATCTACTCCACATCTACACTATTCTTTATCAATACCATTCATTGTAGTCTCTCACTGTCAGCGTCGTCAAGGCCTCTCCCTACTTTTCTTTCTTTTTTTTAACCCCTATGG"]
       
def iProm_Zea(Strng):
    p=0
    n=0
    prediction=""
    X1 = {}
    my_hottie = encode_seq((Strng))
    out_final=my_hottie
    out_final = np.array(out_final)
    X1[accumulator]=out_final
      #out_final=list(out_final)
    X1[accumulator] = out_final    
    X1 = list(X1.items()) 
    an_array = np.array(X1)
    an_array=an_array[:,1]    
    transpose = an_array.T
    transpose_list = transpose.tolist()
    X1=np.transpose(transpose_list)
    X1=np.transpose(X1)
    pr=modelProMN.predict(X1)
    return pr


predictions=[]
def predict_seq(seqs):
    for i in range(len(seqs)):
        pred=iProm_Zea(seqs[i])
        predictions.append(pred)
    return predictions



Testsequences = [] 
for record in SeqIO.parse("D:/Research papers/iPromZea/IProm_Zea Submitted/Webserver And Dataset/IProm-Zea Dataset/Dataset/Training & Testing/First Layer/Z_Mays_first_layer_n_test.txt", "fasta"):
    if(len(record.seq)==251):
        Testsequences.append(record.seq.upper())
TestSeq=[]
for i in range (len(Testsequences)):
    c=Testsequences[i]
    b=c._data
    b=b.decode(encoding="utf-8")
    b=b[:251]
    if(len(b)==251):
        #newNeg.append(b)
        TestSeq.append(b)

pr2= predict_seq(TestSeq)
pr2=np.array(pr2)

pr3=pr2[:3]
for r in pr2:
   for c in r:
       maxxi=max(c)
       if(c[0]==maxxi):
        print("TATA Class")
       elif(c[1]==maxxi):
        print("TATA-Less Class")
       elif(c[2]==maxxi):
        print("Non Promoter")
