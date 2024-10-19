****Iris Species Prediction App****

**App WebLink:**  https://predict-iris-species-xcq4wdhovse5tbenfghfk3.streamlit.app/

**Overview**
This project is a Streamlit-based web application that predicts the species of the famous Iris dataset using three machine learning algorithms:

Logistic Regression

Support Vector Machine (SVM)

K-Means Clustering

The app also provides users with the ability to visualize the dataset in 3D and interact with the model by making predictions on custom input data.

![image](https://github.com/user-attachments/assets/033d3fd8-f971-47b7-ae23-d9c5266e99b5)


**Features**

**Data Display:** View the Iris dataset and a detailed description report.

**Model Selection:** Choose between Logistic Regression, SVM, and K-Means Clustering.

**Hyperparameter Tuning:** Customize model-specific hyperparameters using the sidebar.

**Model Training and Evaluation:** Train the model and display evaluation metrics like accuracy, precision, confusion matrix, and classification report.

**Visualization:** View 3D scatter plots of the dataset and an Elbow plot for K-Means.

**Custom Predictions:** Predict the species based on user-specified flower measurements (Sepal and Petal dimensions).

**Libraries Used**

Streamlit: For building the web app.

Pandas: For data manipulation and analysis.

NumPy: For numerical computations.

Plotly Express: For generating interactive visualizations.

Scikit-learn: For machine learning algorithms, including Logistic Regression, SVM, and K-Means.

**Instructions**

**1. Installation**

Make sure you have Python installed. You can install the required libraries by running:


**pip install streamlit pandas numpy plotly scikit-learn**

**2. Running the Application**
Once the dependencies are installed, you can run the Streamlit app with the following command:

streamlit run app.py

This will open the app in your web browser.

**3. Using the Application**

Model Selection: Choose a model (Logistic Regression, SVM, or K-Means) from the sidebar.

![image](https://github.com/user-attachments/assets/0feaf799-d99c-4330-a2c8-87afebe7d9f9)


Hyperparameter Tuning: Set the hyperparameters for the selected model using the sidebar options.

Show Data: View the Iris dataset with species information.

Evaluation Metrics: After training the model (for Logistic Regression and SVM), the app displays key metrics like accuracy and precision, as well as the confusion matrix and classification report.

**Visualization:** Generate a 3D scatter plot of the Iris dataset, and an Elbow plot for K-Means clustering.

**Custom Predictions:** Use the sliders to input custom values for Sepal and Petal dimensions, and the app will predict the corresponding Iris species.

![image](https://github.com/user-attachments/assets/83648622-e17c-43b1-95ed-7ef9a7b063c4)


**3D Scatter Plot:** Visualizes the Iris dataset with dimensions like Sepal length, Sepal width, and Petal width.

![image](https://github.com/user-attachments/assets/c06a92c9-f7f8-4adb-866a-3ec7fed566ad)


**Evaluation Metrics:**

Accuracy: Provides the overall accuracy of the model.

Precision: Displays the precision score for the model.

Confusion Matrix: A matrix showing true vs. predicted values.

Classification Report: A detailed report showing precision, recall, f1-score, and support for each class (species).

![image](https://github.com/user-attachments/assets/473a0437-1aeb-40e7-8973-668a11824938)


Custom Prediction: Based on user input (Sepal/Petal dimensions), the app predicts the Iris species and displays the result.
