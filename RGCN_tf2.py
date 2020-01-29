# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:38:50 2019

@author: xiaoweijia
"""

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random

###### Declare constants ######
learning_rate = 0.01
learning_rate_pre = 0.005
epochs = 100
epochs_pre = 200#70
batch_size = 2000
hidden_size = 20 
input_size = 20-2
phy_size = 2
T = 13149
cv_idx = 2 # 0,1,2 for cross validation, 4383 lenth for each one
length = 4383
npic = 12
n_steps = int(length/npic) # cut it to 16 pieces #43 #12 #46 
n_classes = 1 
N_sec = (npic-1)*2+1
N_seg = 42
kb=1.0

###### build model ######

class rgcn(layers.Layer):

    def __init__(self, hidden_units, input_dim):
        super().__init__()
        self.W2 = tf.Variable([hidden_size, n_classes], tf.float32,
                                         tf.random_normal_initializer(stddev=0.02))
        self.b2 = tf.Variable([n_classes],  tf.float32,
                                         initializer=tf.constant_initializer(0.0))

        self.Wg1 = tf.Variable([hidden_size, hidden_size], tf.float32,
                                         tf.random_normal_initializer(stddev=0.02))
        self.bg1 = tf.Variable([hidden_size],  tf.float32,
                                         initializer=tf.constant_initializer(0.0))
        self.Wg2 = tf.Variable([hidden_size, hidden_size], tf.float32,
                                         tf.random_normal_initializer(stddev=0.02))
        self.bg2 = tf.Variable([hidden_size],  tf.float32,
                                         initializer=tf.constant_initializer(0.0))


        self.Wa1 = tf.Variable([hidden_size, hidden_size], tf.float32,
                                         tf.random_normal_initializer(stddev=0.02))
        self.Wa2 = tf.Variable([hidden_size, hidden_size], tf.float32,
                                         tf.random_normal_initializer(stddev=0.02))
        self.ba = tf.Variable([hidden_size],  tf.float32,
                                         initializer=tf.constant_initializer(0.0))

        self.Wc1 = tf.Variable([hidden_size, hidden_size], tf.float32,
                                         tf.random_normal_initializer(stddev=0.02))
        self.Wc2 = tf.Variable([hidden_size, hidden_size], tf.float32,
                                         tf.random_normal_initializer(stddev=0.02))
        self.bc = tf.Variable([hidden_size],  tf.float32,
                                         initializer=tf.constant_initializer(0.0))


        self.Wp = tf.Variable([hidden_size, phy_size], tf.float32,
                                         tf.random_normal_initializer(stddev=0.02))
        self.bp = tf.Variable([phy_size],  tf.float32,
                                         initializer=tf.constant_initializer(0.0))


