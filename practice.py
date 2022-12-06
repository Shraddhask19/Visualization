import streamlit as st
import time
import numpy as np
import pandas as pd


# st.write("Hello ,let's learn how to build a streamlit app together")
# st.title ("this is the app title")
# st.header("this is the markdown")
# st.markdown("this is the header")
# st.subheader("this is the subheader")
# st.caption("this is the caption")
# st.code("x=2021")
# st.latex(r''' a+a r^1+a r^2+a r^3 ''')


# st.image("kid.jpg")
# st.audio("Audio.mp3")
# st.video("video.mp4")



# st.checkbox('yes')
# st.button('Click')
# st.radio('Pick your gender',['Male','Female'])
# st.selectbox('Pick your gender',['Male','Female'])
# st.multiselect('choose a planet',['Jupiter', 'Mars', 'neptune'])
# st.select_slider('Pick a mark', ['Bad','Normal', 'Good', 'Excellent'])
# st.slider('Pick a number', 0,50)



# st.number_input('Pick a number', 0,10)
# st.text_input('Email address')
# st.date_input('Travelling date')
# st.time_input('School time')
# st.text_area('Description')
# st.file_uploader('Upload a photo')
# st.color_picker('Choose your favorite color')



# st.balloons()
# st.progress(10)
# with st.spinner('Wait for it...'):
#     time.sleep(2)




# st.success("You did it !")
# st.error("Error")
# st.warning("Warning")
# st.info("It's easy to build a streamlit app")
# st.exception(RuntimeError("RuntimeError exception"))


uploaded_file = st.file_uploader("Choose a file")
# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.write(df)
# df = pd.read_csv(uploaded_file)
# dataframe = pd.DataFrame(uploaded_file,
#   columns = ('col %d' % i
#     for i in range(10)))
# st.write(df)  


c1,c2 = st.columns(2)
st.write(uploaded_file)
data = pd.read_csv("Iris_Data_Sample (1).csv")
ndata = data.iloc[:,:-1]  
with c1:
    st.subheader("Mean")
    # for col in (ndata.columns):
    #     sum=0
    #     for i in range(len(ndata)):
    #         if(type(ndata.loc[i, col])!=type("navjyot")):
    #             sum = sum + ndata.loc[i, col]
    #     mean = sum/len(ndata)
    #     st.write(col)  
c1.write("this")
# st.write(data.head(0))