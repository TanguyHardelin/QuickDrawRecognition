#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf

class QuickDrawRecognitionModel(tf.keras.Model):
    
    def __init__(self):
        super(QuickDrawRecognitionModel, self).__init__()

        self.layer_list = []

        #Convolutionnal layers:
        self.layer_list.append(tf.keras.layers.ZeroPadding2D((1,1)))
        self.layer_list.append(tf.keras.layers.Conv2D(64, 3, activation="relu"))
        self.layer_list.append(tf.keras.layers.ZeroPadding2D((1,1)))
        self.layer_list.append(tf.keras.layers.Conv2D(64, 3, activation="relu"))

        self.layer_list.append(tf.keras.layers.ZeroPadding2D((1,1)))
        self.layer_list.append(tf.keras.layers.Conv2D(128, 3, activation="relu"))
        self.layer_list.append(tf.keras.layers.ZeroPadding2D((1,1)))
        self.layer_list.append(tf.keras.layers.Conv2D(128, 3, activation="relu"))
        self.layer_list.append(tf.keras.layers.MaxPool2D((3,3), strides=(3,3)))

        self.layer_list.append(tf.keras.layers.ZeroPadding2D((1,1)))
        self.layer_list.append(tf.keras.layers.Conv2D(256, 3, activation="relu"))
        self.layer_list.append(tf.keras.layers.MaxPool2D((3,3), strides=(3,3)))

        self.layer_list.append(tf.keras.layers.ZeroPadding2D((1,1)))
        self.layer_list.append(tf.keras.layers.Conv2D(512, 3, activation="relu"))
        self.layer_list.append(tf.keras.layers.MaxPool2D((3,3), strides=(3,3)))

        self.layer_list.append(tf.keras.layers.Dropout(0.5))
        self.layer_list.append(tf.keras.layers.Dense(4096,activation="relu"))
        self.layer_list.append(tf.keras.layers.Dropout(0.5))
        self.layer_list.append(tf.keras.layers.Dense(4096,activation="relu"))
        self.layer_list.append(tf.keras.layers.Dense(10,activation="softmax"))
        
    def call(self,x):

        for layer in self.layer_list:
            x=layer(x)

        return x




