# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

# Title of the app
st.title("Risk Model Deployment with Streamlit")

# Sidebar for user inputs
st.sidebar.header('Input Values for Prediction')

# Predefined categorical options (you can adjust these according to your dataset)
gender_options = ['Male', 'Female']
car_options = ['Yes', 'No']
property_options = ['Yes', 'No']
employment_options = ['Employed', 'Unemployed', 'Self-employed', 'Other']
education_options = ['High School', 'Bachelors', 'Masters', 'PhD']
marital_options = ['Single', 'Married', 'Divorced', 'Widowed']
dwelling_options = ['Own', 'Rent', 'Other']

# Collecting user inputs
gender = st.sidebar.selectbox('Gender', gender_options)
has_car = st.sidebar.selectbox('Has a Car', car_options)
has_property = st.sidebar.selectbox('Has a Property', property_options)
employment_status = st.sidebar.selectbox('Employment Status', employment_options)
education_level = st.sidebar.selectbox('Education Level', education_options)
marital_status = st.sidebar.selectbox('Marital Status', marital_options)
dwelling = st.sidebar.selectbox('Dwelling Type', dwelling_options)
age = st.sidebar.slider('Age', 18, 100, 30)
children_count = st.sidebar.slider('Children Count', 0, 10, 2)
income = st.sidebar.number_input('Income', value=50000)
employment_length = st.sidebar.slider('Employment Length (years)', 0, 40, 5)
has_work_phone = st.sidebar.selectbox('Has a Work Phone', ['Yes', 'No'])
has_email = st.sidebar.selectbox('Has an Email', ['Yes', 'No'])
account_age = st.sidebar.slider('Account Age (years)', 0, 40, 10)

# Converting inputs to a dataframe
input_data = pd.DataFrame({
    'Gender': [gender],
    'Has a car': [has_car],
    'Has a property': [has_property],
    'Employment status': [employment_status],
    'Education level': [education_level],
    'Marital status': [marital_status],
    'Dwelling': [dwelling],
    'Age': [age],
    'Children count': [children_count],
    'Income': [income],
    'Employment length': [employment_length],
    'Has a work phone': [has_work_phone],
    'Has an email': [has_email],
    'Account age': [account_age]
})

# Display input data
st.subheader('User Input Data')
st.write(input_data)

# Encoding categorical variables
le = LabelEncoder()
for col in ['Gender', 'Has a car', 'Has a property', 'Employment status',
            'Education level', 'Marital status', 'Dwelling', 'Has a work phone', 'Has an email']:
    input_data[col] = le.fit_transform(input_data[col])

# Loading the trained RandomForest model and PCA
best_rf_model = joblib.load('best_rf_model_compressed.joblib')
pca = joblib.load('pca_compressed.joblib')

# Scaling the input data
scalar = StandardScaler()
input_data_scaled = scalar.fit_transform(input_data)

# Applying PCA to the input data
input_data_pca = pca.transform(input_data_scaled)

# Making a prediction
if st.button('Predict'):
    prediction = best_rf_model.predict(input_data_pca)
    prediction_proba = best_rf_model.predict_proba(input_data_pca)

    # Display the prediction result
    if prediction[0] == 1:
        st.write("The model predicts: **High Risk**")
    else:
        st.write("The model predicts: **Low Risk**")

    # Display prediction probabilities
    st.write(f"Prediction Probability: Low Risk = {prediction_proba[0][0]:.2f}, High Risk = {prediction_proba[0][1]:.2f}")

# Feature importance plot
st.subheader('Feature Importances')
features = ['Gender', 'Has a car', 'Has a property', 'Employment status',
            'Education level', 'Marital status', 'Dwelling', 'Age', 'Children count', 
            'Income', 'Employment length', 'Has a work phone', 'Has an email', 'Account age']

importances = best_rf_model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
st.pyplot(plt)
