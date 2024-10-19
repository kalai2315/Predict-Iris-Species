import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score

# Load the iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target).map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

st.title('Predict Iris Species')
st.write('This app predicts the Iris species!')

show_data = st.button('Show Data')
if show_data:
    st.dataframe(X.join(y.rename('species')))
    
st.subheader('Data Information')
st.write('Number of Rows:', X.shape[0])
st.write('Number of Columns:', X.shape[1])

st.subheader('Description Report')
st.dataframe(X.describe())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

Model = st.sidebar.selectbox('Select Model/Algorithm', ('Logistic Regression', 'SVM', 'K-Means'))

# Hyperparameters configuration
def hyperparameters(Model):
    params = {}
    if Model == 'Logistic Regression':
        solver = st.sidebar.selectbox('Solver', ('sag', 'saga'))
        max_iter = st.sidebar.slider('Max Iterations', 50, 200)
        params['solver'] = solver
        params['max_iter'] = max_iter

    elif Model == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        kernel = st.sidebar.selectbox('Kernel', ('linear', 'poly', 'rbf', 'sigmoid'))
        params['C'] = C
        params['kernel'] = kernel

    elif Model == 'K-Means':
        n_clusters = st.sidebar.slider('Number of Clusters', 1, 10)
        params['n_clusters'] = n_clusters

    return params

params = hyperparameters(Model)

# Create the model based on the selected algorithm
def classifier(Model, params):
    if Model == 'Logistic Regression':
        clf = LogisticRegression(**params)
    elif Model == 'SVM':
        clf = SVC(**params)
    elif Model == 'K-Means':
        clf = KMeans(**params)
    return clf

clf = classifier(Model, params)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)

if Model != 'K-Means':
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    st.header('Evaluation Metrics')
    st.subheader('Accuracy Score')
    acc = accuracy_score(y_test, y_pred)
    st.write(f'Accuracy: {acc}')

    st.subheader('Precision Score')
    prec = precision_score(y_test, y_pred, average='macro')
    st.write(f'Precision: {prec}')

    st.subheader('Confusion Matrix')
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

    st.subheader('Classification Report')
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

else:
    clf.fit(X)
    st.write('Model trained with K-Means Clustering.')

# Visualization section
def visualize_data(Model):
    st.header('Visualization')
    with st.spinner('Generating Visualizations...'):
        data_with_species = pd.DataFrame(iris.data, columns=iris.feature_names)
        data_with_species['species'] = y
        fig = px.scatter_3d(data_with_species, 
                            x='sepal length (cm)', 
                            y='sepal width (cm)', 
                            z='petal width (cm)', 
                            color='species', 
                            size='petal length (cm)', 
                            title='3D Scatter Plot of Iris Dataset')
        st.plotly_chart(fig)

        if Model == 'K-Means':
            scores = [KMeans(n_clusters=i+1).fit(X).inertia_ for i in range(1, 11)]
            no_of_clusters = np.arange(1, 11)
            fig1 = px.line(x=no_of_clusters, y=scores, title='Elbow Plot')
            fig1.update_layout(xaxis_title='Number of Clusters', yaxis_title='Scores')
            st.plotly_chart(fig1)

visualize_data(Model)

# Prediction on custom data
def custom_prediction():
    st.header('Predict Species on Custom Data')
    sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0)
    sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5)
    petal_length = st.slider('Petal Length (cm)', 1.0, 7.0)
    petal_width = st.slider('Petal Width (cm)', 0.1, 3.0)

    custom_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = clf.predict(custom_data)
    st.write('Predicted Species:', le.inverse_transform(prediction))

custom_prediction()
