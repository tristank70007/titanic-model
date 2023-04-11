# Download the libraries needed: scikit-learn, numpy and pandas
# e.g. pip install scikit-learn
        
 # Importing the necessary libraries for the model

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import os

# Reading in data

test_data = pd.read_csv("C:/Users/Tristan Kelly/Desktop/Python/Data/Titanic Data/test.csv")
training_data = pd.read_csv("C:/Users/Tristan Kelly/Desktop/Python/Data/Titanic Data/train.csv")


# The next few sections of code is to analyze, and clean the data
# The predictors being used for the original model are ['Pclass', 'Sex', 'Age', 'Parch', 'Fare']
# We can start by cleaning up these fields and the response variable (survival) and readdress other fields if they are brought in in the future

# Get a high level view of the test data

print("High Level View: \n")
print(training_data.head())
print(training_data.tail())


# Returning the null value counts per column

null_test_counts = training_data.isnull().sum()
print(null_test_counts)


# Using get_dummies() to create binary columns for the 'Sex' column
# Still don't know why the sex column with strings is causing an error

training_data = pd.get_dummies(training_data, columns=['Sex'])
print(training_data.head())


# Print the unique values of the different columns

#for column in test_data.columns:
 #   print(test_data[column].unique())


# Target Object

y = training_data.Survived


# Input Data

features = ['Pclass', 'Sex_female', 'Sex_male', 'Parch', 'Fare', 'SibSp']
X = training_data[features]



# Separate validation and training data 

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state= 0)


# Define Model

titanic_model = RandomForestClassifier(random_state = 0)


# Fit the model

titanic_model.fit(train_X, train_y)


# Initial Predictions for the training model

training_prediction = titanic_model.predict(val_X)
print("The prediction for the training data is: {}".format(training_prediction))


# Accuracy Score for the training model

training_accuracy = round(accuracy_score(val_y, training_prediction),2)
print("Accuracy Score: {}%".format(training_accuracy))

# Initial Mean Absolute Error of the training model

test_mae = round(mean_absolute_error(training_prediction, val_y),2)
print("Mean Absolute Error is: {}%".format(test_mae))


# Before working on tuning the model we are going to test the model with the test data

 