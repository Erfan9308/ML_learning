import pandas as pd
# To read the data sets more easily

import numpy as np
#Allows for a arrays ( we usually use arrays not lists)

import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle


data = pd.read_csv("student_mat_2173a47420.csv", sep=";")

# Since our data is seperated by semicolons we need to do sep=";"

print(data.head())  # This will print out the first 5 students in our data frame.

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]] # We don't need all the atributes in our data frame, only these six atributes

predict = "G3"

X = np.array(data.drop([predict], axis=1)) # Features
y = np.array(data[predict]) # Labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])