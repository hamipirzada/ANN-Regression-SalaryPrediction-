import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load the trained model
model = tf.keras.models.load_model('regression_model.h5')

# Load the encoder and scaler
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit App
st.title('Estimated Salary Prediction')

st.subheader("Customer Information")
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0], key="geo")
gender = st.selectbox('Gender', label_encoder_gender.classes_, key="gender")
age = st.slider('Age', 18, 92, key="age")
credit_score = st.number_input('Credit Score', key="credit_score")
balance = st.number_input('Balance', key="balance")
tenure = st.slider('Tenure', 0, 10, key="tenure")
num_of_products = st.slider('Number of Products', 1, 4, key="num_of_products")
has_cr_card = st.selectbox('Has Credit Card', [0, 1], key="has_cr_card")
is_active_member = st.selectbox('Is Active Member', [0, 1], key="is_active_member")
exited = st.selectbox('Exited', [0, 1])
# Prepare the input data

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with the input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Prediction Churn
prediction = model.predict(input_data_scaled)
predicted_salary = prediction[0][0]

# Display the result
st.write("Predicted Estimated Salary: {:.2f}".format(predicted_salary))