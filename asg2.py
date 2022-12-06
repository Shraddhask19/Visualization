import streamlit as st
import time
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import math
import seaborn as sns
from scipy import stats
from numpy import cov
from scipy.stats import pearsonr

global uploaded_file

def app():
    st.title("Data Pre-Processing")
    uploaded_file = st.file_uploader("Choose a file")
    # print(uploaded_file)
    st.write("---")
    def cstest():
        st.subheader("Chi-Square Test")
        data = pd.read_csv(uploaded_file.name)
        ndata = data.iloc[:,:-1]
        arr = ndata.columns
        cnt=1
        c1,c2 = st.columns(2)
        
        with c1:
            option1 = st.selectbox(label='Attribute 1',options=(arr),key=cnt)
            cnt += 1
        
        with c2:
            option2 = st.selectbox(label='Attribute 2',options=(arr),key=cnt)
            cnt += 1 

        # print(data[option1].value_counts())
        # print(data[option2].value_counts())
        st.write('#')
        st.write("Contingency Table")
        ct=pd.crosstab(data[option1], data[option2], margins=True)
        st.write(ct)

        obs = np.append(ct.iloc[0][0:4].values, ct.iloc[1][0:4].values)
        row_sum = ct.iloc[0:2,4].values
        exp = []
        for j in range(2):
            for val in ct.iloc[2,0:4].values:
                exp.append(val * row_sum[j] / ct.loc['All', 'All'])
        chi_sq_stats = ((obs - exp)**2/exp).sum()
        dof = (len(row_sum)-1)*(len(ct.iloc[2,0:4].values)-1)

        st.write('---')

        c3,c4 = st.columns(2)

        st.write('---')

        with c3:
            st.write("Calculated value without using scipy")
            st.write("Chi-Square Value: ", chi_sq_stats)
            st.write("Degrees Of Freedom: ",dof)
        
        obs = np.array([ct.iloc[0][0:4].values,ct.iloc[1][0:4].values])

        with c4:
            st.write("Calculated value by using scipy")
            st.write("Chi-Square Value: ", stats.chi2_contingency(obs)[0:1])
            st.write("Degrees of freedom: ", stats.chi2_contingency(obs)[2:3])

    cstest()



    def pearson():
        data = pd.read_csv(uploaded_file.name)
        arr = data.columns
        cnt=10

        st.subheader("Pearson's Coefficient and Covariance")

        c1,c2 = st.columns(2)
        with c1:
            option1 = st.selectbox(label='Attribute 1',options=(arr),key=cnt)
            cnt += 1
        
        with c2:
            option2 = st.selectbox(label='Attribute 2',options=(arr),key=cnt)
            cnt += 1 
        
        sum = 0
        for i in range(len(data)):
            sum += data.loc[i, option1]
        avg1 = sum/len(data)
        sum = 0
        for i in range(len(data)):
            sum += (data.loc[i, option1]-avg1)*(data.loc[i, option1]-avg1)
        var1 = sum/(len(data))
        sd1 = math.sqrt(var1)
        
        sum = 0
        for i in range(len(data)):
            sum += data.loc[i, option2]
        avg2 = sum/len(data)
        sum = 0
        for i in range(len(data)):
            sum += (data.loc[i, option2]-avg2)*(data.loc[i, option2]-avg2)
        var2 = sum/(len(data))
        sd2 = math.sqrt(var2)
        
        sum = 0
        for i in range(len(data)):
            sum += (data.loc[i, option1]-avg1)*(data.loc[i, option2]-avg2)
        covariance = sum/len(data)
        pearsonCoeff = covariance/(sd1*sd2)    

        st.write('#')

        c3,c4 = st.columns(2)
        
        with c3:
            st.write("Calculation without using inbuilt libraries")
            st.write("The Covariance is: ",covariance)
            st.write("The Pearson's Coefficient is: ",pearsonCoeff)
        
        with c4:
            st.write("Calculation with using inbuilt libraries")
            st.write("The Covariance is: ",cov(data[option1],data[option2])[1][0])
            st.write("The Pearson's Coefficient is: ",pearsonr(data[option1], data[option2])[0])


        if pearsonCoeff > 0:
            c3.write("Attributes " + option1 + ' and ' + option2 + " are positively correlated.")
        elif pearsonCoeff < 0:
            c3.write("Attributes " + option1 + ' and ' + option2 + " are negatively correlated.")
        elif pearsonCoeff == 0:
            c3.write("Attributes " + option1 + ' and ' + option2 + " are independant.")

        st.write("---")
    
    pearson()

    def norm():
        st.subheader("Normalization Techniques")
        # st.write('#')
        data = pd.read_csv(uploaded_file.name)
        arr = data.columns
        cnt=20

        c1,c2,c3 = st.columns(3)

        with c1:
            # c4,c5=st.columns(2)
            option = st.selectbox(label='Normalization Techniques',options=["Min-Max normalization", 
            "Z-Score normalization", "Normalization by decimal scaling"],key=cnt)
            cnt += 1

        with c2:
            option1 = st.selectbox(label="Attribute 1",options=arr,key=cnt)
            cnt+=1

        with c3:
            option2 = st.selectbox(label="Attribute 2",options=arr,key=cnt)
            cnt+=1


        if option == "Min-Max normalization":
            n = len(data)
            arr1 = []
            for i in range(len(data)):
                arr1.append(data.loc[i, option1])
            arr1.sort()
            min1 = arr1[0]
            max1 = arr1[n-1]
            
            arr2 = []
            for i in range(len(data)):
                arr2.append(data.loc[i, option2])
            arr2.sort()
            min2 = arr2[0]
            max2 = arr2[n-1]
            
            for i in range(len(data)):
                data.loc[i, option1] = ((data.loc[i, option1]-min1)/(max1-min1))
            
            for i in range(len(data)):
                data.loc[i, option2] = ((data.loc[i, option2]-min2)/(max2-min2))
        elif option == "Z-Score normalization":
            sum = 0
            for i in range(len(data)):
                sum += data.loc[i, option1]
            avg1 = sum/len(data)
            sum = 0
            for i in range(len(data)):
                sum += (data.loc[i, option1]-avg1)*(data.loc[i, option1]-avg1)
            var1 = sum/(len(data))
            sd1 = math.sqrt(var1)
            
            sum = 0
            for i in range(len(data)):
                sum += data.loc[i, option2]
            avg2 = sum/len(data)
            sum = 0
            for i in range(len(data)):
                sum += (data.loc[i, option2]-avg2)*(data.loc[i, option2]-avg2)
            var2 = sum/(len(data))
            sd2 = math.sqrt(var2)
            
            for i in range(len(data)):
                data.loc[i, option1] = ((data.loc[i, option1]-avg1)/sd1)
            
            for i in range(len(data)):
                data.loc[i, option2] = ((data.loc[i, option2]-avg2)/sd2)
        elif option == "Normalization by decimal scaling":        
            j1 = 0
            j2 = 0
            n = len(data)
            arr1 = []
            for i in range(len(data)):
                arr1.append(data.loc[i, option1])
            arr1.sort()
            max1 = arr1[n-1]
            
            arr2 = []
            for i in range(len(data)):
                arr2.append(data.loc[i, option2])
            arr2.sort()
            max2 = arr2[n-1]
            
            while max1 > 1:
                max1 /= 10
                j1 += 1
            while max2 > 1:
                max2 /= 10
                j2 += 1
            
            for i in range(len(data)):
                data.loc[i, option1] = ((data.loc[i, option1])/(pow(10,j1)))
            
            for i in range(len(data)):
                data.loc[i, option2] = ((data.loc[i, option2])/(pow(10,j2)))

            st.write("Normalized values")
        i = 0

        # c4,c5,c6,c7 = st.columns(4)
        st.dataframe(data=data)

        butPlot = st.button(label="Scatter Plot", key=cnt)
        if butPlot:
            sns.scatterplot(x=option1, y=option2, data=data)
            st.pyplot()

    norm()