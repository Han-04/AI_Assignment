import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# --- Let the user choose the model ---
model_choice = st.sidebar.selectbox("Choose a Model", ["ANN", "SVM", "KNN"])

# --- Load the selected model and the preprocessor ---
try:
    if model_choice == "ANN":
        model = load_model('ann_model.keras')
    elif model_choice == "SVM":
        model = joblib.load('svm_churn_model.joblib')
    else: # Default to KNN
        model = joblib.load('knn_churn_model.joblib')
    
    preprocessor = joblib.load('preprocessor.joblib')

except FileNotFoundError:
    st.error("Model or preprocessor file not found.")
    st.stop()

# --- Streamlit App Interface ---
st.title("Bank Customer Churn Prediction")
st.markdown("Enter customer details to predict whether they will churn or not.")

# --- Create Input Fields in the Sidebar ---
st.sidebar.header("Customer Input Features")

# Create a dictionary to hold the input data
input_data = {}

# Get user input for each feature the model expects
# NOTE: The order and names must match the training data before preprocessing
input_data['CreditScore'] = st.sidebar.number_input('Credit Score', min_value=300, max_value=850, value=650)
input_data['Geography'] = st.sidebar.selectbox('Geography', ['France', 'Spain', 'Germany'])
input_data['Gender'] = st.sidebar.selectbox('Gender', ['Male', 'Female'])
input_data['Age'] = st.sidebar.slider('Age', 18, 92, 38)
input_data['Tenure'] = st.sidebar.slider('Tenure (years)', 0, 10, 5)
input_data['Balance'] = st.sidebar.number_input('Account Balance', value=75000.0, format="%.2f")
input_data['NumOfProducts'] = st.sidebar.slider('Number of Products', 1, 4, 1)
input_data['IsActiveMember'] = st.sidebar.selectbox('Is Active Member?', [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
# Add any other features your final model uses
# For example, if you kept 'Satisfaction Score':
input_data['Satisfaction Score'] = st.sidebar.slider('Satisfaction Score', 1, 5, 3)


# --- Prediction Logic ---
if st.sidebar.button("Predict"):
    # Convert the input data into a DataFrame
    # This is crucial as the preprocessor expects a DataFrame with correct column names
    input_df = pd.DataFrame([input_data])

    # Pre-process the input data using the loaded preprocessor
    processed_input = preprocessor.transform(input_df)

    # Make the prediction
    if model_choice == "ANN":
        # Keras .predict() returns the probability of the positive class (churn)
        prediction_proba = model.predict(processed_input)[0][0]
        # Convert the probability into a binary class label (0 or 1)
        prediction = 1 if prediction_proba > 0.5 else 0
        
        # Assign probabilities for display
        prob_churn = prediction_proba
        prob_retain = 1 - prediction_proba
    else:
        # Scikit-learn .predict() returns the class label directly
        prediction = model.predict(processed_input)[0]
        # .predict_proba() returns probabilities for both classes
        prediction_proba = model.predict_proba(processed_input)[0]

        # Assign probabilities for display
        prob_churn = prediction_proba[1] # Probability of churn (class 1)
        prob_retain = prediction_proba[0] # Probability of retention (class 0)

    # --- Display the Prediction (This part is now universal) ---
    st.subheader(f"Prediction Result (using {model_choice} model)")

    if prediction == 1:
        st.error("This customer is likely to CHURN.")
        st.write(f"Confidence (Churn Probability): {prob_churn*100:.2f}%")
    else:
        st.success("This customer is likely to be RETAINED.")
        st.write(f"Confidence (Retention Probability): {prob_retain*100:.2f}%")