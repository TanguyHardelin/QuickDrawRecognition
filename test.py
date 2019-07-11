#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

from models import QuickDrawRecognitionModel
from utils import decrypt_command_line

import sys
import cv2



#display help
if(len(sys.argv)<2):
    help_file = open("help_files/test_help.txt","r")
    help_text = help_file.read()
    print(help_text)

command = decrypt_command_line.decrypt_command_line(sys.argv)
model_name       = command["--name"]
image_name       = command["--image"]

model = QuickDrawRecognitionModel.QuickDrawRecognitionModel()

oriimg = cv2.imread(image_name,cv2.COLOR_BGR2GRAY)
newimg = cv2.resize(oriimg,(28,28))


model.load_weights("checkpoints/"+model_name+".tf")
print(model.predict(newimg))
    



