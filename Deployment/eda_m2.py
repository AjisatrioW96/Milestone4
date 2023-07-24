import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
from scipy.stats import mode


st.set_page_config(
    page_title = 'Customer Churn Prediction',
    page_icon="🧊",
    initial_sidebar_state='expanded',
        menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)


colors = sns.color_palette('PuBu')
colors_1 = sns.color_palette('PuBu')[4:7]
colors_2 = sns.color_palette('PuBu')[2:7]
colors_3 = sns.color_palette('PuBu')[1:7]
colors_4 = ['#7209B7', '#4895EF', '#560BAD', '#480CA8', '#4361EE' ]


def run(): 

    #Membuat Title
    st.title('Customer Churn Prediction')

    #Membuat sub header
    st.subheader('Exploratory Data for Customer Churn Prediction')

    image = Image.open('dataset-cover.png')
    st.image(image, caption = 'Churn', use_column_width ='auto')


    c1, c2 = st.columns((5,5))
    with c1: 
        st.write('Customer churn refers to the phenomenon where customers discontinue their relationship or subscription with a company or service provider. It represents the rate at which customers stop using a companys products or services within a specific period. Churn is an important metric for businesses as it directly impacts revenue, growth, and customer retention')
    with c2:
        st.write('In the context of the Churn dataset, the churn label indicates whether a customer has churned or not. A churned customer is one who has decided to discontinue their subscription or usage of the companys services. On the other hand, a non-churned customer is one who continues to remain engaged and retains their relationship with the company.')





    st.write('#### Dataset of Churn Prediction')
    data = pd.read_csv('customer_churn_dataset-training-master.csv')
    data.dropna(inplace=True)
    st.dataframe(data)


    
    st.write('#### Exploratory Data of Churn Prediction')
    st.markdown("##")
    st.sidebar.header("Please Filter Here:")

    # Update the session state variables when the user selects an option
    selected_tep1 = st.sidebar.selectbox(
        "Select to Categorical Columns:", ['Gender', 'Subscription Type' , 'Contract Length' ]
    ) 
    
    selected_tep2 = st.sidebar.selectbox(
        "Select to Numerical Columns:", ['Age', 'Tenure' , 'Usage Frequency', 'Support Calls', 'Payment Delay' , 'Total Spend' , 'Last Interaction' ] 
    )

    sns.set_theme()
    

    def plot_pie(variables , m ) :
        if m[variables].nunique() == 2 :
            explodes = [0, 0.05]
        else :   
            explodes = [0, 0.05, 0.025]
        m[variables].value_counts().plot.pie(autopct='%1.0f%%', 
                                    pctdistance=0.7,
                                    explode = explodes,
                                    wedgeprops = {'edgecolor' : 'black'},  
                                    labeldistance=1.1, 
                                    colors=colors_4 )
        plt.title(f'{variables} Distribution')


        
    def eda_countplot(x, y, data):
        feat = x
        hue = y
        hue_type = data[hue].dtype.type
        groups = data[feat].unique()
        proportions = data.groupby(feat)[hue].value_counts(normalize=True).unstack()
        ax = sns.countplot(x=feat, hue=hue, data=data, palette =colors_4)
        for c in ax.containers:
            labels = [f'{proportions.loc[g, hue_type(c.get_label())]:.1%}' for g in groups]
            ax.bar_label(c, labels)
        ax = plt.subplots(facecolor='None')
        
    

    d1, d2 = st.columns((5,7))
    with d1 :
        st.markdown(f"<h2 style='text-align: center; font-size: 20px;'>Pieplot {selected_tep2} Distribution</h2>", unsafe_allow_html=True)
        fig, ax = plt.subplots(facecolor='None')
        fig.set_size_inches(5, 4)
        plot_pie(selected_tep1 , data)
        st.pyplot(fig)
    with d2 :
        st.markdown(f"<h2 style='text-align: center; font-size: 20px;'>Churn CountPlot by {selected_tep1}</h2>", unsafe_allow_html=True)
        fig, ax1 = plt.subplots()
        fig, ax = plt.subplots(facecolor='None')
        fig.set_size_inches(5, 4)
        eda_countplot(selected_tep1, 'Churn', data)
        st.pyplot(fig)

    
    def eda_kdeplot(x, y, data):
        sns.kdeplot(data=data, x=x, hue=y, multiple="stack", palette =colors_4, alpha=.7, fill = True)
        mode_number = data[data[y] == 1][x].value_counts().idxmax()
        median_number = np.median(data[x])
        ax.text(0.7, 0.6, f'Mode {x}: {mode_number:.2f}\nMedian {x}: {median_number:.2f}', transform=ax.transAxes)
        ax.axvline(mode_number, color='red', linestyle='--', linewidth=1)
        handles, labels = ax.get_legend_handles_labels()
        

    
    e1, e2, e3 = st.columns((2, 15, 2))
    with e2 :
        st.markdown(f"<h2 style='text-align: center; font-size: 20px;'>Churn KDEplot by {selected_tep2}</h2>", unsafe_allow_html=True)
        fig, ax1 = plt.subplots()
        fig, ax = plt.subplots(facecolor='None')
        fig.set_size_inches(5, 4)
        eda_kdeplot(selected_tep2, 'Churn', data)
        st.pyplot(fig)




if __name__ == '__main__':
    run()

