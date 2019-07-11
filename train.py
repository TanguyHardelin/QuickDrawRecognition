#!/usr/bin/env python
# coding: utf-8

# # Import


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from models import *

# ## Load dataset:

max_images_per_cathegory = 20000

dataset_dir="datasets"
dataset_files = os.listdir(dataset_dir)
# Count number of drawing
count = 0
for file in dataset_files:
    data  = np.load(os.path.join(dataset_dir,file))
    data  = data[:max_images_per_cathegory]
    count += data.shape[0]

print(count,"images in total")

images          = np.zeros((count,784))
targets         = np.zeros(count)
targets         = targets.astype(np.int)
target_names    = []

i=0;j=0
for file in dataset_files:
    d = np.load(os.path.join(dataset_dir,file))
    d = d[:max_images_per_cathegory]
    images[i:i+d.shape[0]]  = d
    targets[i:i+d.shape[0]] = j
    i+=d.shape[0]
    j+=1
    target_names.append(file.split("full_numpy_bitmap_")[1].split(".npy")[0])

print(target_names)
#plt.imshow(images[10].reshape(28,28))
#plt.title(target_names[targets[10]])
#plt.show()


# ## Normalisation

scalar = StandardScaler()
images = scalar.fit_transform(images.reshape(-1,28*28))
images = images.reshape(-1,28,28,1)
images = images.astype(np.float32)

#shuffle data:
indexes = np.arange(count)
np.random.shuffle(indexes)
images = images[indexes]
targets = targets[indexes]


# ## Split into two datasets
(train_images,test_images,train_targets,test_targets) = train_test_split(images,targets,test_size=0.33)


# ## Convert data to tensor
train_dataset = tf.data.Dataset.from_tensor_slices((train_images,train_targets)).shuffle(11)
test_dataset  = tf.data.Dataset.from_tensor_slices((test_images,test_targets)).shuffle(11)


# # Training



loss_object     = tf.keras.losses.SparseCategoricalCrossentropy()
accuracy_object = tf.keras.metrics.SparseCategoricalAccuracy()
optimizer       = tf.keras.optimizers.Adam()

#Accumulateurs:
train_loss_history     = tf.keras.metrics.Mean()
train_accuracy_history = tf.keras.metrics.Mean()

test_loss_history     = tf.keras.metrics.Mean()
test_accuracy_history = tf.keras.metrics.Mean()

model = models.QuickDrawRecognitionModel.QuickDrawRecognitionModel()

@tf.function
def train_step(images,targets):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(targets,predictions)
    
    gradients = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    
    accuracy = accuracy_object(targets,predictions)

    train_loss_history(loss)
    train_accuracy_history(accuracy)



@tf.function
def test_step(images,targets):
    predictions = model(images)
    loss = loss_object(targets,predictions)
    accuracy = accuracy_object(targets,predictions)
    
    test_loss_history(loss)
    test_accuracy_history(accuracy)


epoch = 25
batch_size = 500

history={"train":{"accuracy":[],"loss":[]},"test":{"accuracy":[],"loss":[]}}

for epoch in range(0,epoch):
    print("\n")
    for images,targets in train_dataset.batch(batch_size):
        train_step(images,targets)
    
    for images,targets in test_dataset.batch(batch_size):
        test_step(images,targets)
    print("Epoch",epoch,"train loss",train_loss_history.result(),"train accuracy",train_accuracy_history.result(),
                        "test loss",test_loss_history.result(),"test accuracy",test_accuracy_history.result())

    #History:
    history["train"]["accuracy"].append(train_accuracy_history.result())
    history["train"]["loss"].append(train_loss_history.result())

    history["test"]["accuracy"].append(test_accuracy_history.result())
    history["test"]["loss"].append(test_loss_history.result())

    train_accuracy_history.reset_states()
    train_loss_history.reset_states()
    test_loss_history.reset_states()
    test_accuracy_history.reset_states()
    
#display training accurancy:
plt.plot(history["train"]["accuracy"])
plt.plot(history["test"]["accuracy"])
plt.title("Accuracy")
plt.show()


plt.plot(history["train"]["loss"])
plt.plot(history["test"]["loss"])
plt.title("Loss")
plt.show()


