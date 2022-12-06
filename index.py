import requests 
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
st.set_page_config(layout="wide")
import asg1
import asg2
import asg3
import asg7
import asg6
import as5
import asg8



def app():
    def load_lottieurl(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    lottie_anim1 = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_VeqtOe.json")


    with st.container():
        st.subheader("Hi, I am Navjyot Sakhalkar :wave:")
        st.title("A Data Miner")
        st.write("This app provides defferent data mining and data visualization tools")
        st.write("[Learn More About Data Mining >](https://www.geeksforgeeks.org/data-mining/)")

    with st.container():
        st.write("---")
        left_col, right_col = st.columns(2)
        with left_col:
            st.header("What this app provides")
            st.write("##")
            st.write(
                """
                Data Mining Tools:
                - Data Analysis Tool
                - Data Preprocessing Tasks
                - Normalization Techniques
                - Classification Tasks
                """
            )
    
        with right_col:
            st_lottie(lottie_anim1, height=350, key="data miner")


selected = option_menu(
    menu_title="",
    options=['Home', 'Data Analysis', 'Data Pre-Processing', 'Decision Tree Classifier', 'Extract Rules', 'Classification', 'Clustering', 'ARM', 'Web Mining'],
    orientation="horizontal"
)

if selected=='Data Analysis':
    asg1.app()
elif selected=='Data Pre-Processing':
    asg2.app()
elif selected=='Decision Tree Classifier':
    asg3.app()
elif selected=='Classification':
    as5.app()
elif selected=='Clustering':
    asg6.app()
elif selected=='ARM':
    asg7.app()
elif selected=='Web Mining':
    asg8.app()
else:
    app()