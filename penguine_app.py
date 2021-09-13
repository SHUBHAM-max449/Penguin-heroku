import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Penguine data classifier
 This app predicts the **palmer penguine** sepcies
 
 Data obtained from [palmerpenguis librery](https://github.com/dataprofessor/code/blob/master/streamlit/part3/penguins_cleaned.csv)
""")

st.sidebar.header("User input features")
upload_file=st.sidebar.file_uploader("Upload file here")
st.sidebar.header("OR")
if upload_file is not None:
    df=pd.read_csv('upload_file')
else:
    def user_input_feature():
        island = st.sidebar.selectbox('island',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('sex',('male','female'))
        bill_length_mm = st.sidebar.slider("Bill length (mm) ",32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider("Bill depth (mm)",13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)',172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider("Body mass",2700.00,6300.00,4207.00)
        data={'island': island,
              'bill_length_mm': bill_length_mm,
              'sex': sex,
              'bill_depth_mm': bill_depth_mm,
              'flipper_length_mm': flipper_length_mm,
              'body_mass_g': body_mass_g}
        feature=pd.DataFrame(data, index=[0])
        return feature
    input_df=user_input_feature()

penguine_raw = pd.read_csv('penguins_cleaned.csv')
penguine = penguine_raw.drop(columns=['species'])
df = pd.concat([input_df,penguine],axis=0)
encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col])
    df = pd.concat([df, dummy], axis=1)
    del df[col]

df = df[:1]
st.subheader("User input feature")
if upload_file is not None:
    st.write(df)
else:
    st.write(df)

load_clf=pickle.load(open('penguine_clc.pkl','rb'))

prediction = load_clf.predict(df)
prediction_prob = load_clf.predict_proba(df)

st.subheader('Prediction')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_prob)









