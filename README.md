# Podstawy Nauki o Danych :computer:
## Introduction
We were given a dataset, which constains "Fasion_MNIST" data. Those are the images of cloths collected by company named - "Zalando". Our main goal is to build a model which can classifies those images well.
We were given two datasets:
  1. Training dataset which contain 60 000 elemnts
  2. Test dataset which constains 10 000 elements
## Methods
Main library which I used is called : SciKit. <br />
Beside that, I use: datatime, Firebase etc.

To download data, I used Firebase Storage in which I uploded data at the begining of the process.
I built a model which train itself using training data. After that is evaluates itself uisng testing data.
After every stage, You are able to see time output which shows how long it took to process.

##### recources:
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html <br />
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html <br />
https://www.youtube.com/watch?v=I1eskLk0exg

Main library which I used is called : TensorFlow and Keras. <br />
Beside that, I use: numpy, ssl etc.

To download data, I use built-in Tensor-Flow module which allows me to download data in easy way.
I built a model which train itself using training data. After that is evaluates itself uisng testing data.
After every stage, You are able to see time output which shows how long it took to process.

##### recources:
https://www.tensorflow.org/guide/keras/sequential_model <br />
https://keras.io/guides/sequential_model/ <br />
https://stackoverflow.com/questions/37891954/keras-how-do-i-predict-after-i-trained-a-model <br />
https://intellij-support.jetbrains.com/hc/en-us/community/posts/360000117730-SSL-error-in-accessing-MNIST-dataset

## Results
### 3_Task
| Main              | First Result  | Second Result | Thirs Result |
| ------------------| ------------- | ------------- | ------------ |
| Neighbour number  | 1             | 5             | 9            |
| Accuracy          | 0.846         | 0.854         | 0.851        |

| Benchmark         | First Result  | Second Result | Third Result |
| ------------------| ------------- | ------------- | ------------ |
| Neighbour number  | 1             | 5             | 9            |
| Accuracy          | 0.847         | 0.860         | 0.856        |

### 4_Task
| Main              | First Result  | Second Result | Third Result | Fourth Result |
| ------------------| ------------- | ------------- | ------------ | ------------- |
| Echos             | 5             | 15            | 50           | 150           |
| Accuracy          | 0.846         | 0.851         | 0.883        | 0.890         |

## Usage
In the Taks_3 and Task_4 the Fasion_Mnist data downloads automaticly. In the Task_3 it uses Firebse to fetch the data from the web, what takes about 30 sec. In the Taks_4 data is downloads automaticly using TensorFlow buil-in method.
To run everything, first You need to install nedded libraries using "requirements.txt" file.
After the installation, You can run Taks_3 or Taks_4 and wait for the results.

(Unfortunetly, now it is not possible to download the ML model.)



