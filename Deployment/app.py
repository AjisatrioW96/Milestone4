import streamlit as st
import eda_m2
import prediction_m2

navigation = st.sidebar.selectbox('Pilih halman :', ('EDA', 'Make a Prediction'))

if navigation == 'EDA' :
    eda_m2.run()
else :
    prediction_m2.run()