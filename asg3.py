import string
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import math
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from six import StringIO
from sklearn import metrics
from IPython.display import Image
import pydotplus
from os import system
from sklearn import tree
from sklearn.datasets import load_iris

# st.title("Decision Tree Classifier")

global uploaded_file

def app():
    global data,uploaded_file,ndata,last_col,x,y
    st.title("Decision Tree Classifier")
    uploaded_file = st.file_uploader("Choose a file")
    st.write("---")
    

    def decisionTree():
        data = pd.read_csv(uploaded_file.name)
        ndata = data.iloc[:,:-1]
        last_col = data.iloc[:,-1]
        column_list = []
        for i in ndata.columns:
            column_list.append(i)
    
        x=data[column_list]
        y=data[last_col.name]

        class_names = []

        with open(uploaded_file.name, 'r') as csv:
            data = [line.strip().split(',')[-1] for line in csv.readlines()]
            list_of_words = set(data)
            for word in list_of_words:
                # if word not in class_names:
                #     class_names.append(str(word))
                count = data.count(word)
                if count==1:
                    continue
                class_names.append(str(word))

        # del class_names[0]
        # st.write(class_names)

        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=1)

        detc = DecisionTreeClassifier(criterion="entropy")
        detc = detc.fit(X_train, y_train)
        y_pred = detc.predict(X_test)
        # print(y_pred)
        dot_data = StringIO()

        export_graphviz(detc, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                    feature_names = column_list, class_names=class_names)

        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

        graph.write_png('iris.png')

        st.image('iris.png')

    def decTree():
        data = pd.read_csv(uploaded_file.name)
        ndata = data.iloc[:,:-1]
        last_col = data.iloc[:,-1]
        column_list = []
        for i in ndata.columns:
            column_list.append(i)
    
        x=data[column_list]
        y=data[last_col.name]

        class_names = []

        with open(uploaded_file.name, 'r') as csv:
            data = [line.strip().split(',')[-1] for line in csv.readlines()]
            list_of_words = set(data)
            for word in list_of_words:
                # if word not in class_names:
                #     class_names.append(str(word))
                count = data.count(word)
                if count==1:
                    continue
                class_names.append(str(word))

        # del class_names[0]
        # st.write(class_names)

        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=1)

        detc = DecisionTreeClassifier(criterion="gini")
        detc = detc.fit(X_train, y_train)
        y_pred = detc.predict(X_test)
        # print(y_pred)
        dot_data = StringIO()

        export_graphviz(detc, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                    feature_names = column_list, class_names=class_names)

        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

        graph.write_png('iris.png')

        st.image('iris.png')



    butDecTree = st.button("Decision Tree Using Information Gain")
    butDecTree1 = st.button("Decision Tree")

    if butDecTree:
        decisionTree()
    elif butDecTree1:
        decTree()

