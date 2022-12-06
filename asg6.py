import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, adjusted_rand_score
# from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler



global uploaded_file

def app():
    st.title("Clustering Algorithms")
    uploaded_file = st.file_uploader("Choose a file")
    # st.write(uploaded_file)
    st.write("---")
    if uploaded_file:
        iris_data = pd.read_csv(uploaded_file.name)

        # st.write(iris_data.head())

        selectOptions = st.selectbox(label='Select clustering technique', options=['Hierarchical', 'k-means', 'k-medoid', 'DBSCAN'])    

        arr = iris_data.columns
        
        iris_data.drop(['Species'],axis=1, inplace=True)
        
        if id in arr:
            iris_data.drop(['Id'],axis=1, inplace=True)    

        # sns.pairplot(iris_data)
        # st.pyplot()

        arr = iris_data.columns

        if selectOptions=='Hierarchical':
            cnt=123
            c1,c2,c3 = st.columns(3)
            
            with c1:
                option1 = st.selectbox(label='Attribute 1',options=(arr),key=cnt)
                cnt += 1
            
            with c2:
                option2 = st.selectbox(label='Attribute 2',options=(arr),key=cnt)
                cnt += 1 
            
            arr1 = [1,2,3,4]

            with c3:
                option3 = st.selectbox(label='Number of clusters',options=(arr1),key=cnt)
                cnt += 1 

            # iris_data = iris_data[[option1, option2]]
            
            cl1, cl2 = st.columns(2)

            with cl1:
                st.subheader("Before clustering")
                sns.scatterplot(x=option1, y=option2, data=iris_data)
                st.pyplot()

            grps = AgglomerativeClustering(n_clusters=option3, affinity='euclidean', linkage='ward')
            grps.fit_predict(iris_data)
            
            with cl2:
                st.subheader("After clustering")
                plt.scatter(iris_data[option1], iris_data[option2], c=grps.labels_, cmap='cool')
                st.pyplot()

            dist_sin = linkage(iris_data, method="single")
            plt.figure(figsize=(18,6))
            dendrogram(dist_sin, leaf_rotation=90)
            plt.xlabel('Index')
            plt.ylabel('Distance')
            plt.suptitle("Dendrogram Single Method", fontsize=18)
            # plt.show()
            st.pyplot()
        
        elif selectOptions=='k-means':
            cnt = 129
            df = pd.read_csv(uploaded_file.name)
            x=df.iloc[:,[0,1,2,3]].values
            arr1 = [1,2,3,4]
            option1 = st.selectbox(label='Number of clusters',options=(arr1),key=cnt)
            cnt += 1
            kmeans = KMeans(n_clusters=option1, init='k-means++', max_iter=300, n_init=10, random_state=0)
            y_kmeans = kmeans.fit_predict(x)

            clm1, clm2 = st.columns(2)

            with clm1:
                st.subheader("Scatter Plot with Clustering")
                plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, c='purple', label = 'Iris-Setosa')
                plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, c='orange', label = 'Iris-Versicolor')
                plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, c='green', label = 'Iris-Virginica')
                plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label = 'Centroids')
                plt.legend()
                st.pyplot()

            with clm2:
                st.subheader("Correlation of features")
                #finding correlation of features 
                correl=df.corr()
                sns.heatmap(correl,annot=True)
                st.pyplot()

            #Label Encoding - for encoding categorical features into numerical ones
            encoder = LabelEncoder()
            df['Species'] = encoder.fit_transform(df['Species'])
            
            if id in arr:
                df.drop(['Id'],axis=1, inplace=True)
            
            #converting dataframe to np array 
            data = df.values 

            X=data [:, 0:5]
            Y= data [: , -1]

            # print(X.shape)
            # print(Y.shape)

            #train-test split = 3:1 

            train_x = X[: 112, ]
            train_y = Y[:112, ]

            test_x = X[112:150, ]
            test_y = Y[112:150, ]

            # print(train_x.shape)
            # print(train_y.shape)
            # print(test_x.shape)
            # print(test_y.shape)

            #KMeans

            kmeans = KMeans(n_clusters=3)
            kmeans.fit(train_x, train_y)

            # training predictions
            train_labels= kmeans.predict(train_x)

            #testing predictions
            test_labels = kmeans.predict(test_x)

            st.subheader("Accuracy")
            st.write("Training Accuracy: ",accuracy_score(train_y, train_labels)*100)
            st.write("Testing Accuracy: ",accuracy_score(test_labels, test_y)*100)
            
            st.write("---")

            st.subheader("Classification Report")
            st.write(classification_report(train_y, train_labels))
            
        # elif selectOptions=='k-medoid':
            # myiris = datasets.load_iris()
            # x = myiris.data
            # y = myiris.target
            # scaler = StandardScaler().fit(x)
            # x_scaled = scaler.transform(x)
            # kMedoids = KMedoids(n_clusters = 3, random_state = 0)
            # kMedoids.fit(x_scaled)
            # y_kmed = kMedoids.fit_predict(x_scaled)

            # kMedoids = KMedoids(n_clusters = 3, random_state = 0)
            # kMedoids.fit(x_scaled)
            # y_kmed = kMedoids.fit_predict(x_scaled)
            # silhouette_avg = silhouette_score(x_scaled, y_kmed)
            # st.write("Silhouette Average: ",silhouette_avg)

            # sample_silhouette_values = silhouette_samples(x_scaled, y_kmed)
            # for i in range(3):
            #     ith_cluster_silhouette_values = sample_silhouette_values[y_kmed == i]
            #     st.write(np.mean(ith_cluster_silhouette_values))

            # sw = []

            # for i in range(2, 11):
            #     kMedoids = KMedoids(n_clusters = i, random_state = 0)
            #     kMedoids.fit(x_scaled)
            #     y_kmed = kMedoids.fit_predict(x_scaled)
            #     silhouette_avg = silhouette_score(x_scaled, y_kmed)
            #     sw.append(silhouette_avg)

            # plt.plot(range(2, 11), sw)
            # plt.title('Silhoute Score')
            # plt.xlabel('Number of clusters')
            # plt.ylabel('SW')      #within cluster sum of squares
            # st.pyplot()
            # plt.show()
            
            # from sklearn import metrics

            # def purity_score(y_true, y_pred):
            #     # compute contingency matrix (also called confusion matrix)
            #     contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
            #     # return purity
            #     return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

            # kmedoids = KMedoids(n_clusters=3, random_state=0).fit(x_scaled)
            # y_kmed = kmedoids.fit_predict(x_scaled)
            # purity_score(y,y_kmed)

            # plt.scatter(x_scaled[y_kmed == 0, 0], x_scaled[y_kmed == 0, 1], s = 100, c = 'red', label = 'C1')
            # plt.scatter(x_scaled[y_kmed == 1, 0], x_scaled[y_kmed == 1, 1], s = 100, c = 'blue', label = 'C2')
            # plt.scatter(x_scaled[y_kmed == 2, 0], x_scaled[y_kmed == 2, 1], s = 100, c = 'green', label = 'C3')
            # plt.scatter(kmedoids.cluster_centers_[:, 0], kmedoids.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Medoid')
            # st.pyplot()
            # plt.legend()


        elif selectOptions=='DBSCAN':
            opc1, opc2, opc3, opc4 = st.columns(4)

            with opc1:
                option1 = st.text_input('Epsilon value: ', '0.5')

            
            with opc2:
                option2 = st.text_input('Minimum number of samples: ', '8')

            
            with opc3:
                option3 = st.selectbox(label='Attribute 1: ', options=arr)

            
            with opc4:
                option4 = st.selectbox(label='Attribute 2: ', options=arr)


            iris = datasets.load_iris()
            X, Y = iris.data[:, [2,3]], iris.target

            st.write("Features : ", iris.feature_names)
            st.write("Target : ", iris.target_names)
            st.write('Dataset Size : ', X.shape, Y.shape)
            
            with plt.style.context("ggplot"):
                plt.scatter(X[:,0],X[:,1], c = Y, marker="o", s=50)
                plt.xlabel(option3)
                plt.ylabel(option4)
                plt.title("Original Data")

            def plot_actual_prediction_iris(X, Y, Y_preds):
                with plt.style.context(("ggplot", "seaborn")):
                    plt.figure(figsize=(17,6))
                    
                    plot1, plot2 = st.columns(2)

                    with plot1:
                        plt.subplot(1,2,1)
                        plt.scatter(X[Y==0,0],X[Y==0,1], c = 'red', s=50)
                        plt.scatter(X[Y==1,0],X[Y==1,1], c = 'green', s=50)
                        plt.scatter(X[Y==2,0],X[Y==2,1], c = 'blue', s=50)
                        plt.xlabel(option3)
                        plt.ylabel(option4)
                        plt.title("Original Data")
                        st.pyplot()

                    with plot2:
                        plt.subplot(1,2,2)
                        plt.scatter(X[Y_preds==0,0],X[Y_preds==0,1], c = 'red', s=50)
                        plt.scatter(X[Y_preds==1,0],X[Y_preds==1,1], c = 'green', s=50)
                        plt.scatter(X[Y_preds==2,0],X[Y_preds==2,1], c = 'blue', s=50)
                        plt.xlabel(option3)
                        plt.ylabel(option4)
                        plt.title("Clustering Algorithm Prediction")
                        st.pyplot()

            db = DBSCAN(eps=float(option1), min_samples=int(option2))
            Y_preds = db.fit_predict(X)
            plot_actual_prediction_iris(X, Y, Y_preds)
            st.write("Performance of algorithm: ", adjusted_rand_score(Y, Y_preds)*100, "%")

    # app()