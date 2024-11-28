import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoder and scaler


# geo_encoder
with open('ohe.pkl', 'rb') as file:
    ohe = pickle.load(file)

# gender encoder
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender= pickle.load(file)

# scaler encoder
with open('scaler.pkl', 'rb') as file:
    scaler= pickle.load(file)

## Straemlit app
 
st.title("Customer Churn Predictions")

# user inputs
geography = st.selectbox('Geography', ohe.categories_[0])
gender  =  st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Product', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0,1])

#prepare the input data
input_data = {
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],    
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],    
    'HasCrCard': [has_cr_card], 
    'IsActiveMember': [is_active_member],    
    'EstimatedSalary': [estimated_salary],
    'Geography': [geography]
}

input_data_df = pd.DataFrame(input_data)

# Converting the input_data labels for geography
geo_ohe = ohe.transform(input_data_df[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_ohe, columns = ohe.get_feature_names_out(['Geography']))

# concate and dropping the geography column
input_data_df = pd.concat([input_data_df.drop('Geography', axis=1), geo_encoded_df], axis=1)


# Scaling the features using the scaler
input_scaled_data = scaler.transform(input_data_df)

# Make prediction
prediction = model.predict(input_scaled_data)

prediction_probability = prediction[0][0]

if prediction_probability >0.5:
    print("The customer is likey to churn")
else:
    print("The customer is not likey to churn")
    