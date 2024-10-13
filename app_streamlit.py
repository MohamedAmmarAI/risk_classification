# file: app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load pre-trained models
pca_model = joblib.load('pca_compressed.joblib')
rf_model = joblib.load('best_rf_model_compressed.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# Initialize Standard Scaler
scaler = StandardScaler()

# Set up the Streamlit app
st.title("Risk Prediction Model")

# Create input fields for the user to enter data
st.header("Enter the Features Below:")

gender = st.selectbox('Gender', ['Male', 'Female'])
has_car = st.selectbox('Has a car?', ['Yes', 'No'])
has_property = st.selectbox('Has a property?', ['Yes', 'No'])
employment_status = st.selectbox('Employment status', ['Employed', 'Unemployed', 'Student', 'Retired'])
education_level = st.selectbox('Education level', ['High School', 'Bachelors', 'Masters', 'PhD'])
marital_status = st.selectbox('Marital status', ['Single', 'Married', 'Divorced', 'Widowed'])
dwelling = st.selectbox('Dwelling', ['House', 'Apartment'])

children_count = st.number_input('Children count', min_value=0, max_value=10, step=1)
income = st.number_input('Income', min_value=0.0)
age = st.number_input('Age', min_value=18, max_value=100)
employment_length = st.number_input('Employment length (in years)', min_value=0, max_value=40)
has_work_phone = st.selectbox('Has a work phone?', ['Yes', 'No'])
has_email = st.selectbox('Has an email?', ['Yes', 'No'])
account_age = st.number_input('Account age (in years)', min_value=0, max_value=40)

# Create a button for prediction
if st.button("Predict"):
    # Preprocess the input
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Has a car': [has_car],
        'Has a property': [has_property],
        'Employment status': [employment_status],
        'Education level': [education_level],
        'Marital status': [marital_status],
        'Dwelling': [dwelling],
        'Children count': [children_count],
        'Income': [income],
        'Age': [age],
        'Employment length': [employment_length],
        'Has a work phone': [has_work_phone],
        'Has an email': [has_email],
        'Account age': [account_age]
    })
    
    # Convert categorical variables using LabelEncoder
    cat_columns = ['Gender', 'Has a car', 'Has a property', 'Employment status',
                   'Education level', 'Marital status', 'Dwelling']

    for col in cat_columns:
        input_data[col] = label_encoder[col].transform(input_data[col])

    # Scale data
    input_data_scaled = scaler.transform(input_data)
    
    # Apply PCA transformation
    input_data_pca = pca_model.transform(input_data_scaled)

    # Make prediction
    prediction = rf_model.predict(input_data_pca)

    # Output prediction result
    if prediction == 1:
        st.success("This individual is at high risk.")
    else:
        st.success("This individual is not at high risk.")
    
    # Optionally: Display feature importance or other model insights
    st.subheader("Feature Importances")
    features = input_data.columns
    importances = rf_model.feature_importances_
    sorted_indices = importances.argsort()

    st.bar_chart(pd.DataFrame({
        'feature': [features[i] for i in sorted_indices],
        'importance': importances[sorted_indices]
    }).set_index('feature'))

