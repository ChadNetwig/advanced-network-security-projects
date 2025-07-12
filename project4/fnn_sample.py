# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 19:43:04 2019

Updated on Wed Jan 29 10:18:09 2020

@author: created by Sowmya Myneni and updated by Dijiang Huang


Author      : Chad Netwig
Modified    : July 4, 2024

Description :
This script is modified to handle separate training and testing datasets as generated
from Task 2. It supports three scenarios (SA, SB, SC) with consistent parameter setups
for training and testing, including batch size, number of epochs, neural network
architecture, and evaluation metrics.

It preprocesses the datasets by loading the training and testing data separately, extracting
features and labels, and then combining the features for one-hot encoding of categorical features
(specified columns [1, 2, 3]) using sklearn's ColumnTransformer, leaving other columns unchanged.
After encoding, the combined dataset is split back into separate training and testing datasets,
followed by feature scaling using StandardScaler. The script then trains a Feedforward Neural
Network (FNN) model using TensorFlow/Keras, evaluates the model performance on the testing dataset,
and visualizes accuracy and loss metrics.

Added a custom callback TestMetrics to capture and print test metrics (loss and accuracy)
after each epoch during training.

All modifications to this implementation are commented throughout with the prepended
tag "CLN:"

"""

########################################
# Part 1 - Data Pre-Processing
#######################################

# To load a dataset file in Python, you can use Pandas. Import pandas using the line below
import pandas as pd
# Import numpy to perform operations on the dataset
import numpy as np

debug = True  # CLN: Used for printing data structs to console for inspection

# CLN: Define the scenarios with corresponding training and testing datasets
scenarios = {
    "SA": {
        "TrainingData": "Training-a1-a3.csv",
        "TestingData": "Testing-a2-a4.csv"
    },
    "SB": {
        "TrainingData": "Training-a1-a2.csv",
        "TestingData": "Testing-a1.csv"
    },
    "SC": {
        "TrainingData": "Training-a1-a2.csv",
        "TestingData": "Testing-a1-a2-a3.csv"
    }
}

# CLN: Choose the scenario
selected_scenario = "SC"  # CLN: Toggle to "SA", "SB", or "SC" based on scenario

# CLN: Set the training and testing data paths based on the selected scenario
TrainingDataPath = ''  # Assuming the CSV files are in the same directory as the Python script
TrainingData = scenarios[selected_scenario]["TrainingData"]
TestingData = scenarios[selected_scenario]["TestingData"]

# Batch Size
BatchSize = 10
# Epoch Size
NumEpoch = 10

# Import dataset.
# Dataset is given in TrainingData variable You can replace it with the file 
# path such as “C:\Users\...\dataset.csv’. 
# The file can be a .txt as well. 
# If the dataset file has header, then keep header=0 otherwise use header=none
# reference: https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/

# CLN: Load training and testing datasets separately
dataset_train = pd.read_csv(TrainingDataPath + TrainingData, header=None)  # CLN: Load the training dataset
dataset_test = pd.read_csv(TrainingDataPath + TestingData, header=None)    # CLN: Load the testing dataset

#########################
# Training pre-processing
#########################

# CLN: Process the training dataset
# Extract features from the training dataset, excluding the last two columns (features)
X_train = dataset_train.iloc[:, 0:-2].values

# CLN: DEBUG - Print the first 5 rows of test data
if debug:
    print("DEBUG: First 5 rows of extracted features from the training dataset X_train:")
    print(X_train[:5])

# Extract the labels from the training dataset (last but two columns)
label_column_train = dataset_train.iloc[:, -2].values

# CLN: DEBUG - Print the unique extracted labels from training data
if debug:
    unique_labels = np.unique(label_column_train)  # Get unique labels
    print("DEBUG: Unique labels from the training dataset:")
    print(unique_labels)

# Initialize an empty list to store binary labels
y_train = []

# Iterate through each label in the training dataset
for i in range(len(label_column_train)):
    # Check if the label is 'normal'; assign 0 for normal, 1 for attack
    if label_column_train[i] == 'normal':
        y_train.append(0)
    else:
        y_train.append(1)

# Convert the list of labels to a numpy array for further processing
y_train = np.array(y_train)

# CLN: DEBUG - Print the first 5 rows of train data
if debug:
    print("DEBUG: First 100 rows of y_train from train data:")
    print(y_train[:100])


#########################
# Testing pre-processing
#########################

# CLN: Process the testing dataset
# Extract features from the testing dataset, excluding the last two columns (features)
X_test = dataset_test.iloc[:, 0:-2].values

# CLN: DEBUG - Print the first 5 rows of test data
if debug:
    print("DEBUG: First 5 rows of extracted features from the testing dataset X_test:")
    print(X_test[:5])

# Extract the labels from the testing dataset (last but two columns)
label_column_test = dataset_test.iloc[:, -2].values

# CLN: DEBUG - Print the unique extracted labels from testing data
if debug:
    unique_labels = np.unique(label_column_test)  # Get unique labels
    print("DEBUG: Unique labels from the testing dataset:")
    print(unique_labels)

# Initialize an empty list to store binary labels
y_test = []

# Iterate through each label in the testing dataset
for i in range(len(label_column_test)):
    # Check if the label is 'normal'; assign 0 for normal, 1 for attack
    if label_column_test[i] == 'normal':
        y_test.append(0)
    else:
        y_test.append(1)

# Convert the list of labels to a numpy array for further processing (establishes ground truth)
y_test = np.array(y_test)

# CLN: DEBUG - Print the first 5 rows of test data
if debug:
    print("DEBUG: First 100 rows of y_test from test data:")
    print(y_test[:100])

# Step 1: Combine features datasets
X_combined = np.concatenate((X_train, X_test), axis=0)

# CLN: Step 2: One-hot encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [1, 2, 3])],    # CLN: Specify columns for one-hot encoding
    remainder='passthrough'                               # CLN: Leave the rest of the columns untouched
)
X_combined_encoded = ct.fit_transform(X_combined)

# CLN: DEBUG - Print the first 5 rows of X_combined_encoded
if debug:
    print("DEBUG: First 5 rows of X_combined_encoded:")
    print(X_combined_encoded[:5])

# CLN: Step 3: Separate encoded datasets
X_train_encoded = X_combined_encoded[:len(X_train)]
X_test_encoded = X_combined_encoded[len(X_train):]

# Perform feature scaling. For ANN you can use StandardScaler, for RNNs recommended is 
# MinMaxScaler. 
# referece: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# https://scikit-learn.org/stable/modules/preprocessing.html
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_encoded = sc.fit_transform(X_train_encoded)  # CLN: Scaling to the range [0,1]
X_test_encoded = sc.transform(X_test_encoded)        # CLN: Apply the same transformation to the testing dataset

########################################
# Part 2: Building FNN
#######################################

# Importing the Keras libraries and packages
#from tenskeras.models import Sequential
#from keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Initialising the ANN
# Reference: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
classifier = Sequential()

# Adding the input layer and the first hidden layer, 6 nodes, input_dim specifies the number of variables
# rectified linear unit activation function relu, reference: https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=len(X_train_encoded[0])))

# Adding the second hidden layer, 6 nodes
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Adding the output layer, 1 node, 
# sigmoid on the output layer is to ensure the network output is between 0 and 1
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN, 
# Gradient descent algorithm “adam“, Reference: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
# This loss is for a binary classification problems and is defined in Keras as “binary_crossentropy“, Reference: https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# CLN: Define a callback to capture test metrics
class TestMetrics(tf.keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_loss = []
        self.test_accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        x, y = self.test_data
        loss, accuracy = self.model.evaluate(x, y, verbose=0)
        self.test_loss.append(loss)
        self.test_accuracy.append(accuracy)
        print(f'\nTesting loss: {loss:.4f} - Testing accuracy: {accuracy:.4f}\n')

# CLN: Instantiate the callback with the test data
test_metrics = TestMetrics((X_test_encoded, y_test))

# Fitting the ANN to the Training set with the callback
# Train the model so that it learns a good (or good enough) mapping of rows of input data to the output classification.
# add verbose=0 to turn off the progress report during the training
# To run the whole training dataset as one Batch, assign batch size: BatchSize=X_train.shape[0]
classifierHistory = classifier.fit(X_train_encoded, y_train, batch_size=BatchSize, epochs=NumEpoch, callbacks=[test_metrics])

# evaluate the keras model for the provided model and dataset
loss, accuracy = classifier.evaluate(X_test_encoded, y_test)  # CLN: Evaluate on the test data

print('Print the loss and the accuracy of the model on the dataset')
print('Loss [0,1]: %.4f' % (loss), 'Accuracy [0,1]: %.4f' % (accuracy))

########################################
# Part 3 - Making predictions and evaluating the model
#######################################

# Predicting the Test set results
y_pred = classifier.predict(X_test_encoded)
y_pred = (y_pred > 0.9)   # y_pred is 0 if less than 0.9 or equal to 0.9, y_pred is 1 if it is greater than 0.9

# CLN: print predicted vs expected
if debug:
    for i in range(5):
        print('DEBUG: %s => %d (expected %d)' % (X_test[i].tolist(), y_pred[i], y_test[i]))

# Making the Confusion Matrix
# [TN, FP ]
# [FN, TP ]
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
print('Print the Confusion Matrix:')
print('[ TN, FP ]')
print('[ FN, TP ]=')
print(cm)

# CLN: Calculate the accuracy
TN, FP, FN, TP = cm.ravel()  # CLN: Unpack the confusion matrix
calc_accuracy = (TP + TN) / (TP + TN + FP + FN)
print('Calculated Accuracy: {:.4f}'.format(calc_accuracy))

# Compare calculated accuracy and Keras evaluation accuracy
# print('Evaluated Accuracy from Keras: {:.4f}'.format(accuracy))
# assert abs(calc_accuracy - accuracy) < 1e-4, "Calculated and Evaluated accuracies do not match!"

# Generate classification report
report = classification_report(y_test, y_pred, target_names=['Normal', 'Attack'])
print('Classification Report:')
print(report)

########################################
# Part 4 - Visualizing
#######################################

# Import matplotlib libraries for plotting the figures.
import matplotlib.pyplot as plt

# Plot the accuracy
print('Plot the accuracy')
plt.plot(classifierHistory.history['accuracy'], label='train')  # CLN: Plot training accuracy
plt.plot(test_metrics.test_accuracy, label='test')  # CLN: Plot test accuracy
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.ylim(0.7, 1.0)  # Set y-axis limits to focus on a narrower range
plt.savefig('accuracy_sample.png')
plt.show()

# Plot history for loss
print('Plot the loss')
plt.plot(classifierHistory.history['loss'], label='train')  # CLN: Plot training loss
plt.plot(test_metrics.test_loss, label='test')  # CLN: Plot test loss
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.savefig('loss_sample.png')
plt.show()


