#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf

class QuickDrawRecognitionModel(tf.keras.Model):
    
    def __init__(self):
        super(QuickDrawRecognitionModel, self).__init__()

        self.layer_list = []
        
<<<<<<< HEAD
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
=======
        #Convolutionnal layes:
        self.conv0 = tf.keras.layers.Conv2D(4, 3, activation="relu", name="conv0")
        self.conv1 = tf.keras.layers.Conv2D(8, 3, activation="relu", name="conv1")
        self.conv2 = tf.keras.layers.Conv2D(16, 3, activation="relu", name="conv2")
        self.conv3 = tf.keras.layers.Conv2D(32, 3, activation="relu", name="conv3")
>>>>>>> 04c6ba37f51383ba23eac42cbe343ad820745410
        
        #Flatten layer:
        self.layer_list.append(tf.keras.layers.Flatten())
        
        #Dense & output:
<<<<<<< HEAD
        self.layer_list.append(tf.keras.layers.Dropout(0.5))
        self.layer_list.append(tf.keras.layers.Dense(4096,activation="relu"))
        self.layer_list.append(tf.keras.layers.Dropout(0.5))
        self.layer_list.append(tf.keras.layers.Dense(4096,activation="relu"))
        self.layer_list.append(tf.keras.layers.Dense(10,activation="softmax"))
        
    def call(self,x):

        for layer in self.layer_list:
            x=layer(x)

=======
        self.d1 = tf.keras.layers.Dense(32,activation="relu",name="d1")
        self.out = tf.keras.layers.Dense(10,activation="softmax",name="output")
        
    def call(self,x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.flatten_layer(x)
        x = self.out(x)
>>>>>>> 04c6ba37f51383ba23eac42cbe343ad820745410
        return x




