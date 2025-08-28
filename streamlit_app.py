import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# --- Page Configuration (Optional but Recommended) ---
# Sets the title and icon of the browser tab
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🤖",
    layout="centered"
)

# --- Step 1: Load Your BEST Model (ANN) and the Preprocessor ---
# This part is run only once when the app starts up.
@st.cache_resource
def load_assets():
    """
    Loads the trained model and preprocessor from disk.
    The @st.cache_resource decorator ensures this function is run only once.
    """
    try:
        # Load the Keras ANN model
        model = load_model('ann_model.keras')
        
        # Load the preprocessor that was used for all models
        preprocessor = joblib.load('preprocessor.joblib')
        
        return model, preprocessor
    except Exception as e:
        # Display a user-friendly error message if files are not found
        st.error(f"Error loading model or preprocessor files: {e}")
        st.error("Please ensure 'ann_model.keras' and 'preprocessor.joblib' are in the root directory of the app.")
        return None, None

model, preprocessor = load_assets()

# --- Step 2: Streamlit App Interface ---
st.title("Customer Churn Prediction")
st.markdown("This application uses an **Artificial Neural Network** to predict whether a bank customer is likely to churn (leave the bank).")
st.markdown("Please enter the customer's details in the sidebar to get a prediction.")

# --- Step 3: Create Input Fields in the Sidebar ---
st.sidebar.header("Customer Input Features")

# Create a dictionary to hold the input data from the user
input_data = {}

# Define the input widgets in the sidebar
input_data['CreditScore'] = st.sidebar.number_input('Credit Score', min_value=300, max_value=850, value=650, help="Customer's credit score (300-850)")
input_data['Geography'] = st.sidebar.selectbox('Geography', ['France', 'Spain', 'Germany'])
input_data['Gender'] = st.sidebar.selectbox('Gender', ['Male', 'Female'])
input_data['Age'] = st.sidebar.slider('Age', 18, 92, 38, help="Customer's age in years")
input_data['Tenure'] = st.sidebar.slider('Tenure', 0, 10, 5, help="Number of years the customer has been with the bank")
input_data['Balance'] = st.sidebar.number_input('Account Balance', value=75000.0, format="%.2f")
input_data['NumOfProducts'] = st.sidebar.slider('Number of Products', 1, 4, 1, help="Number of bank products the customer uses")
input_data['IsActiveMember'] = st.sidebar.selectbox('Is Active Member?', [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No', help="Is the customer an active member?")
input_data['Satisfaction Score'] = st.sidebar.slider('Satisfaction Score', 1, 5, 3, help="Customer's satisfaction score (1-5)")

# --- Step 4: Prediction Logic ---
# The 'Predict' button is placed in the sidebar
if st.sidebar.button("Predict Churn"):
    # Check if the model and preprocessor were loaded successfully
    if model is not None and preprocessor is not None:
        # Convert the input data dictionary into a Pandas DataFrame
        input_df = pd.DataFrame([input_data])

        # Pre-process the input data using the loaded preprocessor
        processed_input = preprocessor.transform(input_df)

        # Make the prediction using the Keras model
        # The .predict() method returns a 2D array of probabilities, so we get the specific value
        prediction_proba = model.predict(processed_input)[0][0]
        
        # Convert the probability into a binary class label (0 for Retain, 1 for Churn)
        prediction = 1 if prediction_proba > 0.5 else 0

        # --- Step 5: Display the Prediction ---
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("Prediction: **This customer is likely to CHURN.**")
            # Display the probability of churn
            st.write(f"**Confidence (Churn Probability):** `{prediction_proba*100:.2f}%`")
        else:
            st.success("Prediction: **This customer is likely to be RETAINED.**")
            # Calculate and display the probability of retention
            prob_retain = 1 - prediction_proba
            st.write(f"**Confidence (Retention Probability):** `{prob_retain*100:.2f}%`")
    else:
        st.error("Cannot make a prediction because the model assets could not be loaded.")