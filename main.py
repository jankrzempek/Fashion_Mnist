import tensorflow as tf
import requests
import numpy as np
import ssl
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').disabled = True

# WHAT LOGIT MEANS?
# Tensorflow "with logit": It means that you are applying a softmax function to logit numbers to normalize it.
# The input_vector/logit is not normalized and can scale from [-inf, inf].

# TensorFlow needs to be in 1.14 version
# When TensorFlow is deprecated numpy must be at v.1.16

# 1. Train model based on training data
# 2. Mark model based on testing data

# 0	T-shirt / top
# 1	Spodnie
# 2	Zjechać na pobocze
# 3	Sukienka
# 4	Płaszcz
# 5	Sandał
# 6	Koszula
# 7	Sneaker
# 8	Torba
# 9	But do kostki

#SSL additiona to make this code works with TensorFlow
requests.packages.urllib3.disable_warnings()
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

# Downloading the dataset using TensorFlow built in option
fashionMnist = tf.keras.datasets.fashion_mnist
(trainImages, trainLabels), (testImages, testLabels) = fashionMnist.load_data()

# Array which contains names of the products which can occur on photos
nameOfElements = ['T-shirt/top',
               'Trouser',
               'Pullover',
               'Dress',
               'Coat',
               'Sandal',
               'Shirt',
               'Sneaker',
               'Bag',
               'Ankle boot']

# scaling values to range 0-1
trainImages = trainImages / 255.0
testImages = testImages / 255.0

# neurons layers
model = tf.keras.models.Sequential()
# converting images from two-dimensional array to one dimensional array 784 pixels
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# first layer called: Dense has 256 nodes
model.add(tf.keras.layers.Dense(256, activation='relu'))
# second layer contains 10 logits(inputs for softmax function)
model.add(tf.keras.layers.Dense(10))


# model complication
# SGD stand for Stochastic Gradient which we use here
model.compile(optimizer='SGD',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train Model
# Feed model
# epochs - iterations on a dataset
model.fit(trainImages, trainLabels, epochs=150)

# Value accuracy
testLoss, accuracyOfTheTest = model.evaluate(testImages, testLabels, verbose=1)
print("------------------------EXPECTED OUTPUT---------------------")
print('Test accuracy output: ', accuracyOfTheTest)

#Data forecasting
modelOfProbability = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = modelOfProbability.predict(testImages)

print("------------------------COMPARING PREDISCTIONS WITH REALITY---------------------")
# Array of predictions
print(predictions[80])
# This is the highest result which we get using our model
print(np.argmax(predictions[80]))
# This is the result we expect
print(testLabels[80])

numberOfPositiv = 0
numberOfNegativ = 0

print("------------------------COMPARING PREDISCTIONS WITH REALITY IN LOOP---------------------")
for number in range(30):
    numberOfLoss = random.randint(0,1000)
    if np.argmax(predictions[numberOfLoss]) == testLabels[numberOfLoss]:
       numberOfPositiv += 1
    else:
       numberOfNegativ += 1
print(f"Porównanie pozytywnych i negatywnych: {numberOfPositiv} <> {numberOfNegativ}")

