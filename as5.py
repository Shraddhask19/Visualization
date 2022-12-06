import streamlit as st
import tensorflow as tf
from tensorflow import keras
import pandas as pd 
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np
from keras.layers import Dense,Activation,Dropout 
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import scipy.stats
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree, preprocessing
from sklearn import metrics
from keras.models import Sequential 
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.tree import _tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import normalize #machine learning algorithm library

def app():
    st.title("Classification")
    file_name=st.file_uploader("Select a file")
    if file_name:
        df=pd.read_csv(file_name)
        colums=df.columns
        lst=['KNN','Linear Regression','Naive Bayes','ANN']
        c1,c2,c3 = st.columns(3)

        with c1:
            targetAttr=st.selectbox("Choose Target Attribute",colums)
        
        with c2:
            splitvalue = st.text_input("Enter Test Split Percentage", 30)

        with c3:
            algo=st.selectbox("Choose algorithm",lst)
        
        svalue=int(splitvalue)/100
        if algo=='KNN':
            features = list(colums)
            features.remove(targetAttr)
            X = df[features]
            Y = df[targetAttr]
            X_train, X_test, y_train, y_test = train_test_split(
                    X, Y, test_size = svalue, random_state = 0)
            K = []
            training = []
            test = []
            scores = {}
            for k in range(1, 40,2):
                clf = KNeighborsClassifier(n_neighbors = k)
                clf.fit(X_train, y_train)
                training_score = clf.score(X_train, y_train)
                test_score = clf.score(X_test, y_test)
                K.append(k)

                training.append(training_score)
                test.append(test_score)
                scores[k] = [training_score, test_score]
            y_pred = clf.predict(X_test)

            c_matrix = confusion_matrix(y_test, y_pred)

            tp = c_matrix[1][1]
            tn = c_matrix[2][2]
            fp = c_matrix[1][2]
            fn = c_matrix[2][1]

            c1,c2 = st.columns(2)

            with c1:
                st.subheader("Scatter Plot:")
                plt.scatter(K, training, color ='red')
                plt.scatter(K, test, color ='green')
                st.pyplot(plt)
            with c2:
                st.subheader("Confusion Matrix:")
                st.write(c_matrix)
                st.subheader("Classification Report:")
                st.code(classification_report(y_test, y_pred))

        if algo=='Linear Regression':
            le = LabelEncoder()
            label = le.fit_transform(df[targetAttr])
            df.drop(targetAttr, axis=1, inplace=True)
            df[targetAttr] = label
            features = list(colums)
            features.remove(targetAttr)
            X = df[features]
            Y = df[targetAttr]
            X_train, X_test, y_train, y_test = train_test_split(
                    X, Y, test_size = svalue, random_state = 0)
            regr = LinearRegression()
            regr.fit(X_train, y_train)
            c1, c2, c3, c4, c5= st.columns(5)
            with c1:
                #Quartile Q3
                st.write('Regression Coefficient : ')
                st.write(regr.coef_)
            with c2:
                st.write('Intercept : ')
                st.write(regr.intercept_)
            # st.text("")
            with c3:
                st.write("Score:")
                st.write(regr.score(X_test, y_test))
                y_pred = regr.predict(X_test)
            # col1, col2= st.columns(2)
            with c4:
                st.write('Mean Squared Error : ')
                st.write(mean_squared_error(y_test, y_pred))
            with c5:
                st.write('Mean Absolute Error : ')
                st.write(mean_absolute_error(y_test, y_pred))
            # st.text("")

        if algo=='Naive Bayes':
            features = list(colums)
            features.remove(targetAttr)
            X = df[features]
            Y = df[targetAttr]
            X_train, X_test, y_train, y_test = train_test_split(
                    X, Y, test_size = svalue, random_state = 0)
            from sklearn.naive_bayes import GaussianNB
            gnb = GaussianNB()
            gnb.fit(X_train, y_train)
            y_pred = gnb.predict(X_test)
            from sklearn import metrics
            st.write("Gaussian Naive Bayes model accuracy:")
            st.write(metrics.accuracy_score(y_test, y_pred))
            c1,c2 = st.columns(2)
            c_matrix = confusion_matrix(y_test, y_pred)

            tp = c_matrix[1][1]
            tn = c_matrix[2][2]
            fp = c_matrix[1][2]
            fn = c_matrix[2][1]

            with c1:
                st.subheader("Confusion Matrix:")
                st.write(c_matrix)
            
            with c2:
                st.subheader("Classification Report:")
                st.code(classification_report(y_test, y_pred))

        if algo=='ANN':
            btsize = st.text_input("Enter Batch Size", 16)
            numEp = st.text_input("Enter Number of EPOCHS", 4)
            le = LabelEncoder()
            label = le.fit_transform(df[targetAttr])
            df.drop(targetAttr, axis=1, inplace=True)
            df[targetAttr] = label
            features = list(colums)
            features.remove(targetAttr)
            X = df[features]
            Y = df[targetAttr]
            X_normalized=normalize(X,axis=0)
            X_train, X_test, y_train, y_test = train_test_split(
                    X_normalized, Y, test_size = svalue, random_state = 0) 
            y_train=np_utils.to_categorical(y_train,num_classes=3)
            y_test=np_utils.to_categorical(y_test,num_classes=3)    
            model=Sequential()
            model.add(Dense(1000,input_dim=4,activation='relu'))
            model.add(Dense(500,activation='relu'))
            model.add(Dense(300,activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(3,activation='softmax'))
            model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
            BATCH_SIZE=int(btsize)
            EPOCHS=int(numEp)
            VALIDATION_SPLIT=0.2
            model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=20,epochs=10,verbose=1)
            prediction=model.predict(X_test)
            length=len(prediction)
            y_label=np.argmax(y_test,axis=1)
            predict_label=np.argmax(prediction,axis=1)

            accuracy=np.sum(y_label==predict_label)/length * 100 
            print("Accuracy of the dataset",accuracy )

            st.subheader("Accuracy of the dataset")
            st.write(accuracy)
            


            
            # plt.scatter(accuracy,X_train, color ='red')
            # st.pyplot(plt)
            # plt.clf()
            # pd["accuracy"].plot(figsize=(8, 5))
            # plt.title("Accuracy improvements with Epoch")
            # st.pyplot(plt)