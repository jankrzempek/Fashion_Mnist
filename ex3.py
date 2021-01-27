import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import pyrebase
import datetime

# CSV data https://www.kaggle.com/zalando-research/fashionmnist
# firebase config data - unique
firebaseConfig = {
    "apiKey": "AIzaSyB9kOZGJP-238bXcc-S5pN4RfUSYqHF2hI",
    "authDomain": "fasionmnist.firebaseapp.com",
    "databaseURL": "https://project-670430765913.firebaseio.com",
    "projectId": "fasionmnist",
    "storageBucket": "fasionmnist.appspot.com",
    "messagingSenderId": "670430765913",
    "appId": "1:670430765913:web:9738be66f0a6fe63f2a248"
};

startDownloading = datetime.datetime.now()
print("---------------DOWNLOADING DATA---------------")

# initializing firebase storage
firebase = pyrebase.initialize_app(firebaseConfig)
storage = firebase.storage()
database = firebase.database()

pathToTestInCloud = "fashion-mnist_test.csv"
storage.child(pathToTestInCloud).download("fashion-mnist_test.csv")

pathToTrainInCloud = "fashion-mnist_train.csv"
storage.child(pathToTrainInCloud).download("fashion-mnist_train.csv")

endDownloading = datetime.datetime.now()
print(endDownloading-startDownloading)

print("---------------READING DATA---------------")
# Paths which is in the cloud will have same name so everything fine
trainData = pd.read_csv(pathToTrainInCloud)
testData = pd.read_csv(pathToTestInCloud)

# only lebals are here
trainingLabels = trainData.label
testLabels = testData.label

# droping labels and scalind data from 0...1
trainingDataWithoutLabel = trainData.drop("label", axis=1) / 255
testDataWithoutLabel = testData.drop("label", axis=1) / 255

# numbers of data given
numberOfTrainingData = 60000
numberOfTestData = 10000

# random choosing indexes od data
indexOfValidation = np.random.choice(list(set(range(numberOfTrainingData))), numberOfTestData, replace=False)
indexOfTraining = list(set(range(numberOfTrainingData)) - set(indexOfValidation))

# iloc is for finding indexes in the list
# creating training data from this random numbers
xTraining = trainingDataWithoutLabel.iloc[indexOfTraining]
yTraining = trainingLabels.iloc[indexOfTraining]

#creating validation data from this random numbers
xValidation = trainingDataWithoutLabel.iloc[indexOfValidation]
yValidation = trainingLabels.iloc[indexOfValidation]

# Training
print("---------------LEARNING DATA---------------")
startLearning = datetime.datetime.now()
modelOfKnn = KNeighborsClassifier(n_neighbors=5, algorithm ='kd_tree')
modelOfKnn.fit(xTraining, yTraining)

print("---------------PREDICTING DATA---------------")
confusion_matrix(modelOfKnn.predict(xTraining), yTraining)
modelOfKnnAccuracy = (modelOfKnn.predict(xTraining) == yTraining).mean()
endLearning = datetime.datetime.now()

print("Time of training:", endLearning-startLearning)
print("TRAINING ACCURACY = ", modelOfKnnAccuracy)

# Validating
print("---------------VALIDATING DATA---------------")
startValidating = datetime.datetime.now()
confusion_matrix(modelOfKnn.predict(xValidation), yValidation)
modelOfKnnValidationAccu = (modelOfKnn.predict(xValidation) == yValidation).mean()
endValidating = datetime.datetime.now()

print("Time of validating:", endValidating-startValidating)
print("Validation accuracy = ", modelOfKnnValidationAccu)


# Attempts

# 1 - neighbors
# Time of training: 0:17:56.632871
# TRAINING ACCURACY =  1.0
# Time of validating: 0:14:25.086644
# VALIDATION ACCURACY =  0.8466







