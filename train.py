#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os


import utils
from utils import decrypt_command_line, training

import sys

#display help
if(len(sys.argv)<2):
    help_file = open("help_files/train_help.txt","r")
    help_text = help_file.read()
    print(help_text)

command = utils.decrypt_command_line.decrypt_command_line(sys.argv)

epoch            = int(command["--nb_epoch"])
batch_size       = int(command["--batch_size"])
display_funtions = True if command["--display"]=="True" else False
model_name       = command["--name"]

training.training(epoch, batch_size, display_funtions, model_name)