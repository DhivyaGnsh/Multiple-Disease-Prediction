import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load pre-trained models
with open('C:/Users/Mukesh/Downloads/parkinsons_model (2).pkl', 'rb') as file:
    parkinsons_model = pickle.load(file)

with open('C:/Users/Mukesh/Downloads/kidney_disease_model.pkl', 'rb') as file:
    kidney_model = pickle.load(file)

with open('C:/Users/Mukesh/Downloads/liver_disease_model (1).pkl', 'rb') as file:
    liver_model = pickle.load(file)

# Streamlit App Title
st.title("Multiple Disease Prediction System")

# Sidebar for disease selection
disease = st.sidebar.selectbox(
    "Select Disease to Predict",
    ("Parkinson's Disease", "Kidney Disease", "Liver Disease")
)

# Function to preprocess input data
def preprocess_input(input_data, scaler=None):
    input_array = np.array(input_data).reshape(1, -1)
    if scaler:
        input_array = scaler.transform(input_array)
    return input_array

# Parkinson's Disease Prediction
if disease == "Parkinson's Disease":
    st.subheader("Parkinson's Disease Prediction")
    st.write("Enter the following parameters:")

    # Input fields
    fo = st.number_input("MDVP:Fo(Hz) (Average vocal fundamental frequency)", value=0.0)
    fhi = st.number_input("MDVP:Fhi(Hz) (Maximum vocal fundamental frequency)", value=0.0)
    flo = st.number_input("MDVP:Flo(Hz) (Minimum vocal fundamental frequency)", value=0.0)

    # Prepare input
    input_data = [fo, fhi, flo]

    # Prediction
    if st.button("Predict Parkinson's Disease"):
        try:
            input_scaled = preprocess_input(input_data)  # Apply scaler if needed
            prediction = parkinsons_model.predict(input_scaled)
            result = "Positive for Parkinson's Disease" if prediction[0] == 1 else "Negative for Parkinson's Disease"
            st.success(f"Prediction: {result}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Kidney Disease Prediction
elif disease == "Kidney Disease":
    st.subheader("Kidney Disease Prediction")
    st.write("Enter the following parameters:")

    # Input fields
    age = st.number_input("Age", value=0)
    bp = st.number_input("Blood Pressure (BP)", value=0)
    sg = st.number_input("Specific Gravity", value=0.0)
    al = st.number_input("Albumin", value=0)
    su = st.number_input("Sugar", value=0)
    bgr = st.number_input("Blood Glucose Random", value=0.0)
    bu = st.number_input("Blood Urea", value=0.0)
    sc = st.number_input("Serum Creatinine", value=0.0)
    pot = st.number_input("Potassium", value=0.0)
    hemo = st.number_input("Hemoglobin", value=0.0)

    # Prepare input (Select only the features used during training)
    # Replace [age, bp, sg] with the actual features used during training
    input_data = np.array([age, bp, sg]).reshape(1, -1)

    # Prediction
    if st.button("Predict Kidney Disease"):
        try:
            prediction = kidney_model.predict(input_data)
            result = "Positive for Kidney Disease" if prediction[0] == 1 else "Negative for Kidney Disease"
            st.success(f"Prediction: {result}")
        except Exception as e:
            st.error(f"An error occurred: {e}")


# Liver Disease Prediction
elif disease == "Liver Disease":
    st.subheader("Liver Disease Prediction")
    st.write("Enter the following parameters:")

    # Input fields
    age = st.number_input("Age", value=0)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    tb = st.number_input("Total Bilirubin", value=0.0)
    db = st.number_input("Direct Bilirubin", value=0.0)
    alkphos = st.number_input("Alkaline Phosphotase", value=0.0)
    sgpt = st.number_input("Alanine Aminotransferase (SGPT)", value=0.0)
    sgot = st.number_input("Aspartate Aminotransferase (SGOT)", value=0.0)
    tp = st.number_input("Total Proteins", value=0.0)
    alb = st.number_input("Albumin", value=0.0)
    agratio = st.number_input("Albumin/Globulin Ratio", value=0.0)

    # Prepare input
    gender_binary = 1 if gender == "Male" else 0  # Encode gender as binary
    input_data = [age, gender_binary, tb, db, alkphos, sgpt, sgot, tp, alb, agratio]

    # Prediction
    if st.button("Predict Liver Disease"):
        try:
            input_scaled = preprocess_input(input_data)  # Apply scaler if needed
            prediction = liver_model.predict(input_scaled)
            result = "Positive for Liver Disease" if prediction[0] == 1 else "Negative for Liver Disease"
            st.success(f"Prediction: {result}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Footer
st.write("This is a prototype for educational purposes. Ensure to validate the models before deployment.")
