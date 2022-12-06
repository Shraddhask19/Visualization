import streamlit as st
import time
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn_qqplot import pplot

global uploaded_file

def app():
    st.title("Data Analysis Techniques")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)
    print(uploaded_file)
    with st.container():
        st.write("The classes and weights are as follows: ")   
        left_col, right_col = st.columns(2)
        def details():   
            with open(uploaded_file.name, 'r') as csv:
                data = [line.strip().split(',')[-1] for line in csv.readlines()]
                list_of_words = set(data)
                for word in list_of_words:
                    count = data.count(word)
                    if count==1:
                        continue
                    with left_col:
                        st.write(str(word))
                    with right_col:
                        st.write(str(count))
                    # print(word, count)
    
    butDetails = st.button("Details")

    if butDetails:
        details()

    st.write("---")


    def mct ():    
        data = pd.read_csv(uploaded_file.name)
        ndata = data.iloc[:,:-1]
        mean = 0.0
        c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
        with c1:
            st.subheader("Attribute")
            for col in ndata.columns:
                st.write(col)

        with c2:
            st.subheader("Mean")
            for col in (ndata.columns):
                sum=0
                for i in range(len(ndata)):
                    if(type(ndata.loc[i, col])!=type("navjyot")):
                        sum = sum + ndata.loc[i, col]
                mean = sum/len(ndata)
                mean = "{:.2f}".format(mean)
                st.write(str(mean))        

        with c3:
            st.subheader("Mode")
            for col in (ndata.columns):
                freq = {}
                for i in range(len(ndata)):
                    freq[ndata.loc[i, col]] = 0
                maxFreq = 0
                maxFreqElem = 0
                for i in range(len(ndata)):
                    if(type(ndata.loc[i, col])==str):
                        break
                    freq[ndata.loc[i, col]] = freq[ndata.loc[i, col]]+1
                    if freq[ndata.loc[i, col]] > maxFreq:
                        maxFreq = freq[ndata.loc[i, col]]
                        maxFreqElem = ndata.loc[i, col]
                maxFreqElem = "{:.2f}".format(maxFreqElem)
                st.write(str(maxFreqElem))

        with c4:
            st.subheader("Median")
            for col in (ndata.columns):
                n = len(ndata)
                i = int(n/2)
                j = int((n/2)-1)
                arr = []
                for i in range(len(ndata)):
                    arr.append(ndata.loc[i, col])
                arr.sort()
                if n%2 == 1:
                    arr[i] = "{:.2f}".format(arr[i])
                    st.write(str(arr[i]))
                else:
                    ans = "{:.2f}".format((arr[i]+arr[j])/2)
                    st.write(str(ans))

        with c5:
            st.subheader("Midrange")
            for col in (ndata.columns):
                n = len(ndata)
                arr = []
                for i in range(len(ndata)):
                    arr.append(ndata.loc[i, col])
                arr.sort()
                ans = "{:.2f}".format((arr[n-1]+arr[0])/2)
                st.write(str(ans))

        with c6:
            st.subheader("Variance")
        
        with c7:
            st.subheader("S.D")

        for col in ndata.columns:
            sum = 0
            for i in range(len(data)):
                sum += data.loc[i, col]
            avg = sum/len(data)
            sum = 0
            for i in range(len(data)):
                sum += (data.loc[i, col]-avg)*(data.loc[i, col]-avg)
            var = sum/(len(data))
            ans = "{:.2f}".format(var)
            c6.write(str(ans))
            ans = "{:.2f}".format(math.sqrt(var))
            c7.write(str(ans))


    butmct = st.button("Measure of Central Tendancy")

    if(butmct):
        mct()
    
    st.write("---")

    def dispersion():
        data = pd.read_csv(uploaded_file.name)
        ndata = data.iloc[:,:-1]
        
        c1,c2,c3,c4,c5,c6,c7 = st.columns(7)

        with c1:
            st.subheader("Attribute")
            st.write("#")
            for col in ndata.columns:
                st.write(col)
                
        with c2:
            st.subheader("Range")
            st.write("#")
            arr = []
            for col in ndata.columns:
                for i in range(len(data)):
                    arr.append(data.loc[i, col])
                arr.sort()
                ans = "{:.2f}".format(arr[len(data)-1]-arr[0])
                st.write(str(ans))
     
        with c3:
            st.subheader("Lower Quartile") 

        with c4:
            st.subheader("Inter Quartile")
        
        with c5:
            st.subheader("Upper Quartile")
        
        with c6:
            st.subheader("Maximum")
            st.write("#")
        
        with c7:
            st.subheader("Maximum")
            st.write("#")
        
        arr=[]     

        for col in ndata.columns:
            for i in range(len(data)):
                arr.append(data.loc[i, col])
            arr.sort()
            ans = "{:.2f}".format((len(arr)+1)/4)
            c3.write(str(ans))
            ans = "{:.2f}".format((len(arr)+1)/2)  
            c4.write(str(ans))  
            ans = "{:.2f}".format(3*(len(arr)+1)/4)
            c5.write(str(ans))  
            c6.write(str(arr[0]))
            c7.write(str(arr[len(data)-1]))


    butdis = st.button("Dispersion of Data")

    if(butdis):
        dispersion()
    
    st.write("---")


    st.header("Plots")

    def plots():
        data = pd.read_csv(uploaded_file.name)
        ndata = data.iloc[:,:-1]
        arr = ndata.columns
        c1,c2 = st.columns(2)
        
        arr2=[]
        cnt = 1
        # while True:
        with c1:
            c1.subheader("Boxplot")
            option1 = st.selectbox(label='Attribute 1',options=(arr),key=cnt)
            cnt += 1
            option2 = st.selectbox(label='Attribute 2',options=(arr),key=cnt)
            cnt += 1    

            sns.boxplot(x=option1, y=option2, data=ndata)
            st.pyplot()

        with c2:
            c2.subheader("Histogram")
            option1 = st.selectbox(label='Attribute 1',options=(arr),key=cnt)
            cnt += 1
            # option2 = st.selectbox(label='Attribute 2',options=(arr),key=cnt)
            # cnt += 1
            # sns.FacetGrid(data, height=4).map(plt.scatter, option1, option2).add_legend()
            st.write("#")
            st.write("#")    
            sns.histplot(x=option1, data=ndata)
            st.pyplot()
            # plt.show()

        st.write("#")

        c3,c4 = st.columns(2)

        with c3:
            c3.subheader("Scatter Plot")
            option1 = st.selectbox(label='Attribute 1',options=(arr),key=cnt)
            cnt += 1
            option2 = st.selectbox(label='Attribute 2',options=(arr),key=cnt)
            cnt += 1
            sns.scatterplot(x=option1, y=option2, data=ndata)
            st.pyplot()
        
        with c4:
            c4.subheader("Quantile-Quantile Plot")
            option1 = st.selectbox(label='Attribute 1',options=(arr),key=cnt)
            cnt += 1
            option2 = st.selectbox(label='Attribute 2',options=(arr),key=cnt)
            cnt += 1
            pplot(x=option1,y=option2,data=ndata,kind='qq')
            st.pyplot()



    plots()
