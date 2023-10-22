# importing the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# loading the dataset
vm = pd.read_csv('modified_useful_dataset.csv')
vm.columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']

# choosing features and target variable into x and y respectively
x = vm.values[:, :11]
y = vm.values[:, 11]

# splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

# feature scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# initializing classifiers
classifiers = {
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "MLP": MLPClassifier()
}

# initializing a dictionary to store accuracy results
accuracy_results = {}

# training and evaluating each classifier in a loop for avoiding code repetition
for clf_name, clf in classifiers.items():
    clf.fit(x_train, y_train)                    # training the model
    y_pred = clf.predict(x_test)                 # predicting the result and storing it in y_pred
    accuracy = accuracy_score(y_test, y_pred)    # calculating accuracy of the classifier
    accuracy_results[clf_name] = accuracy        # adding classifier name and accuracy int the dictionary

# creating an interactive bar chart with hover effects using Plotly
fig = px.bar(x=list(accuracy_results.keys()), y=list(accuracy_results.values()), labels={"x": "Classifiers", "y": "Accuracy"},
             title="Accuracy Comparison of Classifiers")
fig.update_traces(texttemplate='%{y:.2%}', textposition='outside')  # adding hover effect to show accuracy values
fig.show()

# printing the actual accuracies as console output
for clf_name, accuracy in accuracy_results.items():
    print(f"{clf_name}: {accuracy * 100:.2f}%")
