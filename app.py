import numpy as np
import pickle
import streamlit as st

model = pickle.load(open("model.sav", "rb"))


def prediction(data):
    array = np.asarray(data)
    array_reshaped = array.reshape(1, -1)

    result = model.predict(array_reshaped)
    print(result)
    if result[0] == 0:
        return "NOT SURVIVED"
    else:
        return "SURVIVED"


st.title("Prediction of Survival")

Pclass = st.number_input('Enter Pclass')
Sex = st.number_input('Enter Sex (0:"male", 1:"female")')
Age = st.number_input('Enter Age')
SibSp = st.number_input('Enter SibSp')
Parch = st.number_input('Enter Parch')
Fare = st.number_input('Enter Fare')
Embarked = st.number_input('Enter Embarked (0:"S", 1:"C", 2:"Q")')

input_data = [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]
outcome = ''

if st.button("Predict"):
    outcome = prediction(input_data)

st.success(outcome)

