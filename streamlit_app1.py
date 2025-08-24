import streamlit as st
import pandas as pd
# Removed joblib since we are not loading a model

# --- Streamlit App Interface ---
st.title("Bank Customer Churn Prediction (UI Demo)")
st.markdown("Enter customer details below. (Prediction logic is disabled in this demo).")

# --- Create Input Fields in the Sidebar ---
st.sidebar.header("Customer Input Features")

# Create a dictionary to hold the input data
input_data = {}

# Get user input for each feature
input_data['CreditScore'] = st.sidebar.number_input('Credit Score', min_value=300, max_value=850, value=650)
input_data['Geography'] = st.sidebar.selectbox('Geography', ['France', 'Spain', 'Germany'])
input_data['Gender'] = st.sidebar.selectbox('Gender', ['Male', 'Female'])
input_data['Age'] = st.sidebar.slider('Age', 18, 92, 38)
input_data['Tenure'] = st.sidebar.slider('Tenure (years)', 0, 10, 5)
input_data['Balance'] = st.sidebar.number_input('Account Balance', value=75000.0, format="%.2f")
input_data['NumOfProducts'] = st.sidebar.slider('Number of Products', 1, 4, 1)
input_data['IsActiveMember'] = st.sidebar.selectbox('Is Active Member?', [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
input_data['Satisfaction Score'] = st.sidebar.slider('Satisfaction Score', 1, 5, 3)


# --- Display Input Data on Button Click ---
if st.sidebar.button("Show Input Data"):
    # Convert the input data into a DataFrame for nice display
    input_df = pd.DataFrame([input_data])
    
    st.subheader("Current Input Data")
    st.write("The model would make a prediction based on the following information:")
    
    # Display the DataFrame
    st.dataframe(input_df)