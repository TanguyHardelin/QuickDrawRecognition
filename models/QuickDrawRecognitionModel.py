#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf

class QuickDrawRecognitionModel(tf.keras.Model):
    
    def __init__(self):
        super(QuickDrawRecognitionModel, self).__init__()
        
        #Convolutionnal layes:
        self.conv0 = tf.keras.layers.Conv2D(4, 3, activation="relu", name="conv0")
        self.conv1 = tf.keras.layers.Conv2D(8, 3, activation="relu", name="conv1")
        self.conv2 = tf.keras.layers.Conv2D(16, 3, activation="relu", name="conv2")
        self.conv3 = tf.keras.layers.Conv2D(32, 3, activation="relu", name="conv3")
        
        #Flatten layer:
        self.flatten_layer = tf.keras.layers.Flatten(name="flatten")
        
        #Dense & output:
        self.d1 = tf.keras.layers.Dense(32,activation="relu",name="d1")
        self.out = tf.keras.layers.Dense(10,activation="softmax",name="output")
        
    def call(self,x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.flatten_layer(x)
        x = self.out(x)
        return x




