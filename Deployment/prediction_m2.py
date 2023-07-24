import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

#model inferencing

with open ('model_pipeline_best_dt.pkl' , 'rb') as file_10:
    pipeline_best = pickle.load(file_10)


def run() :

    st.write('# Customer Churn Prediction ')
    background_color_3 = "#a66827"
    st.markdown(
                f"""
                <div style="background-color: {background_color_3}; padding: 10px; border-radius: 5px;">
                    <p><strong>This model prediction still needs to be improved in order to enhance its accuracy and reliability. While it provides valuable insights and predictions, it is essential to use the results with caution.</strong></p>
                </div>
                """,
                unsafe_allow_html=True
            )
    st.markdown('')
    with st.form('key=form_prediction_heart'):
        st.write ('### Fill out the Following:')
        gender_options = {'Female': 'Female', 'Male': 'Male'}
        contract_length_options = {'Monthly': 'Monthly', 'Quarterly': 'Quarterly', 'Annual': 'Annual'}
        subscription_type_options = {'Basic': 'Basic', 'Standard': 'Standard', 'Premium': 'Premium'}
        
        Name = st.text_input('Tuliskan nama customer', 'Abdi')
        gender = st.selectbox('Apa Gender Customer yang anda sedang cari', list(gender_options), index=0)
        contract_length = st.radio('Seberapa lama durasi contract customer?', list(contract_length_options), index=0)
        subscription_type = st.radio('Apa jenis langganan customer?', list(subscription_type_options), index=0)
                
        gender_value = gender_options[gender]
        contract_value = contract_length_options[contract_length]
        subscription_value = subscription_type_options[subscription_type]

        age = st.number_input('Berapa umur dari customer ?', min_value=18 ,max_value=100, value=24)
        tenure = st.number_input('Berapa lama tenure dari customer ?', min_value=1,max_value=60, value=30 )
        usage_frequency = st.slider('Seberapa sering customer memakai applikasi ?',  min_value=1,max_value=50, value=25)
        support_calls = st.slider('Seberapa sering customer memanggil support/ support calls ?',  min_value=0,max_value=50, value=0)
        payment_delay = st.slider('Seberapa lama delay pembayaran yang dilakukan customer ?',  min_value=0,max_value=30, value=10)
        total_spend = st.number_input('Seberapa besar total pengeluaran yang dilakukan customer ?',  min_value=100,max_value=2000, value=750)
        last_interaction = st.slider('Kapan terakhir kali customer terlihat memakai applikasi ?',  min_value=1,max_value=30, value=5)

        submitted = st.form_submit_button('Predict')

        #membuat data baru
    data_inf = {
                'Age' : age, 
                'Gender' : gender, 
                'Tenure' : tenure, 
                'Usage Frequency' : usage_frequency, 
                'Support Calls' : support_calls,
                'Payment Delay' : payment_delay,  
                'Subscription Type' : subscription_type, 
                'Contract Length' :contract_length, 
                'Total Spend' : total_spend ,
                'Last Interaction' : last_interaction
            }
    data_inf = pd.DataFrame([data_inf])


    if submitted: 
            #splitting data
        y_pred_inf = pipeline_best.predict(data_inf)
        background_color_1 = "#e63946"
        background_color_2 = "#3d85c6" 
        
        if y_pred_inf == 1:
            st.write(f'###### Halo, berdasarkan dari model yang diberikan Customer dengan nama {Name} mempunyai : ')
            # Set the background color
    # This is a light red color, you can change it to any other color you prefer
            st.markdown(
                f"""
                <div style="background-color: {background_color_1}; padding: 10px; border-radius: 5px; ">
                    <p><strong>HIGH PROBABILITY TO CHURN</strong></p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown('')
            st.markdown(
                f"""
                <div style="background-color: {background_color_2}; padding: 10px; border-radius: 5px;">
                    <p>The training file for the CHURN dataset contains a collection of 440,882 customer records along with their respective features and churn labels. This file serves as the primary resource for training machine learning models to predict customer churn. The algorithm used in the model is based on Decision Tree, which has an accuracy of 99.7294% in predicting whether a customer will churn or not. However, even though the model has high accuracy, it is essential to recognize that it is not entirely flawless. It's important to understand that while the model performs well in terms of accuracy, there could still be room for improvement and potential challenges to consider.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown('')
        else:
            st.write(f'###### Halo, berdasarkan dari model yang diberikan Customer dengan nama {Name} mempunyai : ')
            st.markdown(
                f"""
                <div style="background-color: {background_color_1}; padding: 10px; border-radius: 5px;">
                    <p><strong>LOW PROBABILITY TO CHURN</strong></p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown('')
            st.markdown(
                f"""
                <div style="background-color: {background_color_2}; padding: 10px; border-radius: 5px;">
                    <p>The training file for the CHURN dataset contains a collection of 440,882 customer records along with their respective features and churn labels. This file serves as the primary resource for training machine learning models to predict customer churn. The algorithm used in the model is based on Decision Tree, which has an accuracy of 99.7294% in predicting whether a customer will churn or not. However, even though the model has high accuracy, it is essential to recognize that it is not entirely flawless. It's important to understand that while the model performs well in terms of accuracy, there could still be room for improvement and potential challenges to consider.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown('')

if __name__ == '__main__':
    run()
