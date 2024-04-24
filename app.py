import streamlit as st
import numpy as np
import pandas as pd
import joblib
import re

model = joblib.load('ml_model.joblib')
st.title('Did they survive? :ship:')
# PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
passengerid = st.text_input("Input Passenger ID", '123456')
pclass = st.select_slider("Choose class", ["First", "Second", "Third"])
if pclass == "First":
    pclass = 1
elif pclass == "Second":
    pclass = 2
else:
    pclass = 3
name  = st.text_input("Input Passenger Name", 'John Smith')
sex = st.selectbox("Choose sex", ['male','female'])
age = st.slider("Choose age",0,100)
sibsp = st.slider("Choose siblings",0,10)
parch = st.slider("Choose parch",0,2)
ticket = st.text_input("Input Ticket Number", "12345") 
fare = st.number_input("Input Fare Price", 0,1000)
cabin = st.text_input("Input Cabin", "C52") 
embarked = st.selectbox("Did they Embark?", ['S','C','Q'])



columns = ["PassengerId","Pclass","Name", "Sex", "Age","SibSp", "Parch","Ticket", "Fare", "Cabin","Embarked"]

def predict(): 
    row = np.array([passengerid,pclass,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked]) 
    x = pd.DataFrame([row], columns = columns)

    x['CabinClass'] = x['Cabin'].fillna('M').apply(lambda x: str(x).replace(" ", "")).apply(lambda x: re.sub(r'[^a-zA-Z]', '', x))
    x['CabinNumber'] = x['Cabin'].fillna('M').apply(lambda x: str(x).replace(" ", "")).apply(lambda x: re.sub(r'[^0-9]', '', x)).replace('', 0) 
    x = x.drop(['PassengerId', 'Name', 'Ticket','Cabin'], axis=1)

    print(x)

    prediction = model.predict(x)
    if prediction[0] == 1: 
        st.success('Passenger Survived :thumbsup:')
    else: 
        st.error('Passenger did not Survive :thumbsdown:') 

trigger = st.button('Predict', on_click=predict)

