#!/usr/bin/env python
# coding: utf-8

# # Import

# In[7]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ## Load dataset:

# In[4]:


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


# In[5]:


images          = np.zeros((count,784))
targets         = np.zeros(count)
target_names    = []

i=0;j=0
for file in dataset_files:
    d = np.load(os.path.join(dataset_dir,file))
    d = d[:max_images_per_cathegory]
    images[i:i+d.shape[0]]  = d
    targets[i:i+d.shape[0]] = j
    i+=d.shape[0]
    j+1
    target_names.append(file.split("full_numpy_bitmap_")[1].split(".npy")[0])

print(target_names)
plt.imshow(images[0].reshape(28,28))
plt.title("Test images")
plt.show()


# ## Normalisation

# In[6]:


scalar = StandardScaler()
images = scalar.fit_transform(images.reshape(-1,28*28))


# ## Split into two datasets

# In[18]:


(train_images,test_images,train_targets,test_targets) = train_test_split(images,targets,test_size=0.33)


# ## Convert data to tensor

# In[19]:


train_dataset = tf.data.Dataset.from_tensor_slices((train_images,train_targets))
test_dataset  = tf.data.Dataset.from_tensor_slices((test_images,test_targets))


# # Training

# In[11]:


loss_object     = tf.keras.losses.SparseCategoricalCrossentropy()
accuracy_object = tf.keras.metrics.SparseCategoricalAccuracy()
optimizer       = tf.keras.optimizers.Adam()

#Accumulateurs:
train_loss_history     = tf.keras.metrics.Mean()
train_accuracy_history = tf.keras.metrics.Mean()

test_loss_history     = tf.keras.metrics.Mean()
test_accuracy_history = tf.keras.metrics.Mean()

#TODO:
#define model here !
model = tf.keras.Sequential()


# In[12]:


@tf.function
def train_step(images,targets):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(targets,predictions)
        
    gradients = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(gradients,model.trainable_variables)
    
    accuracy = accuracy_object(targets,predictions)
    
    train_loss_history(loss)
    train_accuracy_history(accuracy)


# In[13]:


@tf.function
def test_step(images,targets):
    predictions = model(images)
    loss = loss_object(targets,predictions)
    accuracy = accuracy_object(targets,predictions)
    
    test_loss_history(loss)
    test_accuracy_history(accuracy)


# In[27]:


epoch = 100
batch_size = 500

history={"train":{"accuracy":[],"loss":[]},"test":{"accuracy":[],"loss":[]}}
"""
for epoch in range(0,1):
    for images,targets in train_dataset.batch(batch_size):
        #do some things
    
    for images,targets in test_dataset.batch(batch_size):
        #do some things
"""


# In[ ]:




