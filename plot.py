import streamlit as st
import time
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import math
import seaborn as sns


def app():
    data = pd.read_csv('Iris_Data_Sample (1).csv')
    ndata = data.iloc[:,:-1]
    arr = ndata.columns
    c1,c2 = st.columns(2)

    arr2=[]
    cnt = 1
    # while True:
    # with c1:
    option1 = st.sidebar.selectbox(label='Attribute 1',options=(arr))
        # cnt = cnt+1

    
    # with c2:
    option2 = st.sidebar.selectbox(label='Attribute 2',options=(arr))
        # cnt = cnt+1

    # st.write('You selected:', option1," : ",option2)

    sns.relplot(x=option1, y=option2, data=ndata)
    st.pyplot()

app()