import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import sys
sys.path.append('../')
from models import QuickDrawRecognitionModel

#Create model
model = QuickDrawRecognitionModel.QuickDrawRecognitionModel()

## Training
loss_object     = tf.keras.losses.SparseCategoricalCrossentropy()
accuracy_object = tf.keras.metrics.SparseCategoricalAccuracy()
optimizer       = tf.keras.optimizers.Adam()

#Accumulateurs:
train_loss_history     = tf.keras.metrics.Mean()
train_accuracy_history = tf.keras.metrics.Mean()

test_loss_history     = tf.keras.metrics.Mean()
test_accuracy_history = tf.keras.metrics.Mean()

def prepare_datasets():
    ### Load dataset:
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

    ### Normalisation

    scalar = StandardScaler()
    images = scalar.fit_transform(images.reshape(-1,28*28))
    images = images.reshape(-1,28,28,1)
    images = images.astype(np.float32)

    #shuffle data:
    indexes = np.arange(count)
    np.random.shuffle(indexes)
    images = images[indexes]
    targets = targets[indexes]


    ### Split into two datasets
    (train_images,test_images,train_targets,test_targets) = train_test_split(images,targets,test_size=0.33)


    ### Convert data to tensor
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images,train_targets)).shuffle(11)
    test_dataset  = tf.data.Dataset.from_tensor_slices((test_images,test_targets)).shuffle(11)

    return (train_dataset,test_dataset)


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

def training(epoch, batch_size, display_funtions, model_name):
    (train_dataset,test_dataset) = prepare_datasets()

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
    
    if(display_funtions):
        #display training accurancy:
        plt.plot(history["train"]["accuracy"])
        plt.plot(history["test"]["accuracy"])
        plt.title("Accuracy")
        plt.show()

        #display training losses:
        plt.plot(history["train"]["loss"])
        plt.plot(history["test"]["loss"])
        plt.title("Loss")
        plt.show()

    if(model_name):
        
        model.save_weights("checkpoints/"+model_name+".model")

    import cv2
    oriimg = cv2.imread("apple.jpg")
    oriimg = cv2.cvtColor(oriimg, cv2.COLOR_BGR2GRAY)
    newimg = cv2.resize(oriimg,(28,28))
    print(newimg.shape)
    newimg = newimg.astype(np.float32)
    newimg /= 255

    newimg = newimg.reshape(-1,28,28,1)
    #plt.imshow(newimg)
    #plt.show()
    

    print(model(newimg))

