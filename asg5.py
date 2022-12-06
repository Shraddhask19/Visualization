import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns

def app():
    st.title("Classification")
    uploaded_file = st.file_uploader("Choose a file")
    # print(uploaded_file)
    st.write("---")

    df = pd.read_csv(uploaded_file.name)
    # df = df.drop("Id", axis = 1)

    train = df.sample(frac = 0.7, random_state = 1)
    test = df.drop(train.index)

    y_train = train["Species"]
    x_train = train.drop("Species", axis = 1)

    y_test = test["Species"]
    x_test = test.drop("Species", axis = 1)


    means = train.groupby(["Species"]).mean() # Estimate mean of each class, feature
    var = train.groupby(["Species"]).var() # Estimate variance of each class, feature
    prior = (train.groupby("Species").count() / len(train)).iloc[:,1] # Estimate prior probabilities
    classes = np.unique(train["Species"].tolist()) # Storing all possible classes

    def Normal(n, mu, var):
        sd = np.sqrt(var)
        pdf = (np.e ** (-0.5 * ((n - mu)/sd) ** 2)) / (sd * np.sqrt(2 * np.pi))
    
        return pdf

    def Predict(X):
        Predictions = []
        
        for i in X.index: # Loop through each instances
            
            ClassLikelihood = []
            instance = X.loc[i]
            
            for cls in classes: # Loop through each class
                
                FeatureLikelihoods = []
                FeatureLikelihoods.append(np.log(prior[cls])) # Append log prior of class 'cls'
                
                for col in x_train.columns: # Loop through each feature
                    
                    data = instance[col]
                    
                    mean = means[col].loc[cls] # Find the mean of column 'col' that are in class 'cls'
                    variance = var[col].loc[cls] # Find the variance of column 'col' that are in class 'cls'
                    
                    Likelihood = Normal(data, mean, variance)
                    
                    if Likelihood != 0:
                        Likelihood = np.log(Likelihood) # Find the log-likelihood evaluated at x
                    else:
                        Likelihood = 1/len(train) 
                    
                    FeatureLikelihoods.append(Likelihood)
                    
                TotalLikelihood = sum(FeatureLikelihoods) # Calculate posterior
                ClassLikelihood.append(TotalLikelihood)
                
            MaxIndex = ClassLikelihood.index(max(ClassLikelihood)) # Find largest posterior position
            Prediction = classes[MaxIndex]
            Predictions.append(Prediction)
            
        return Predictions

    predicted = Predict(x_test)

    def Accuracy(y, prediction):
        y = list(y)
        prediction = list(prediction)
        score = 0
        
        for i, j in zip(y, prediction):
            if i == j:
                score += 1
                
        return score / len(y)

    def withLibrary():
        clf = GaussianNB() # Gaussian Naïve Bayes assumes gaussian data
        clf.fit(x_train, y_train)
        SkTrain = clf.predict(x_train) # Predicting on the train set
        SkTest = clf.predict(x_test) # Predicting on the test set
        st.write(Accuracy(predicted,SkTest))

    butPred = st.button(label="Prediction", key=12)
    butAcc = st.button(label="Accuracy", key=13)
    butLib = st.button(label="With Library", key=14)

    arr = ["x_test", "y_test", "x_train", "y_train"]
    
    
    if butPred:
        attribute1 = st.selectbox(label="Select class to predict", options=arr,key=15)
        st.write(Predict(attribute1))
    if butAcc:
        st.write(Accuracy(y_test,predicted))
    if butLib:
        st.write(withLibrary())


