#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 15:52:17 2022

@author: loganwoolfson
"""

import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Iris Flower Prediction App

This app predicts the **Iris flower** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 6.1) # slider(text, min, max, default value)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.2)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 3.95)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 1.3)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0]) # best practice to pass data into dict then into dataframe
    return features

df = user_input_features() #assign df to features variable, not sure why made a function for this. maybe easier traceability

st.subheader('User Input parameters')
st.write(df) # displays the dataframe on webpage

#loading dataset into predictors and output
iris = datasets.load_iris()
X = iris.data
Y = iris.target # using index instead of catefory names for training

#fitting the model using all data
clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df) # predict the index of plant type (0,1,2)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction]) # this displays the name of plant for the index
#st.write(prediction) # this displays the index of prediction

st.subheader('Prediction Probability')
st.write(prediction_proba)






