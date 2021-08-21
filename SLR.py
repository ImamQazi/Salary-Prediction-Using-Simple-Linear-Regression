# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 16:19:39 2021

@author: Imam Qazi
"""
#Simple Linear Regression
 
#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing data set
dataset = pd.read_csv('Salary_Data.csv')
#x = dataset['YearsExperience']
#y = dataset['Salary']

#or
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Split Train and Test dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 1/3, random_state = 0)

# Fiting the SLR to Training Set
from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(x_train, y_train)

# Predicting the Test set results
prediction = slr.predict(x_test)

#Visualling the Training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train,slr. predict(x_train), color = 'yellow')
plt.title("Salary Vs Experience Training Set")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualling the Test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train,slr. predict(x_train), color = 'yellow')
plt.title("Salary Vs Experience Training Set")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()