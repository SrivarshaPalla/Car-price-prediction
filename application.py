import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model and data
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')

# Streamlit app title and description
st.title('Car Price Prediction')
st.markdown("This app predicts the price of a used car based on its specifications.")

# Sidebar inputs
st.sidebar.header("Enter Car Specifications")

# Company selection
companies = sorted(car['company'].unique())
companies.insert(0, 'Select Company')
company = st.sidebar.selectbox('Company', companies)

# Car model selection
if company != 'Select Company':
    car_models = sorted(car[car['company'] == company]['name'].unique())
else:
    car_models = sorted(car['name'].unique())
car_model = st.sidebar.selectbox('Car Model', car_models)

# Year selection
year = st.sidebar.selectbox('Year', sorted(car['year'].unique(), reverse=True))

# Fuel type selection
fuel_type = st.sidebar.selectbox('Fuel Type', car['fuel_type'].unique())

# Kilometers driven input
driven = st.sidebar.number_input('Kilometers Driven', min_value=0, max_value=500000, step=1000, value=10000)

# Prediction button
if st.sidebar.button('Predict Price'):
    if company != 'Select Company' and car_model and year and fuel_type and driven:
        # Prepare the input data
        input_data = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                  data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5))
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Display prediction
        st.write(f"### The predicted price of the car is: ₹ {np.round(prediction[0], 2)}")
    else:
        st.write("Please fill out all fields to get a prediction.")

# Additional Information
st.sidebar.header("About")
st.sidebar.text("This app uses a pre-trained Linear Regression model to predict car prices based on input features.")

