import streamlit as st
import numpy as np
import joblib

# Load trained model, encoders, and scaler
model = joblib.load("trained_svm_model.pkl")
sc = joblib.load("scaler.pkl")
le1 = joblib.load("le_sex.pkl")
le2 = joblib.load("le_chest_pain.pkl")
le6 = joblib.load("le_resting_ecg.pkl")
le8 = joblib.load("le_exercise_angina.pkl")
le10 = joblib.load("le_st_slope.pkl")

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Prediction"])

# Home Page
if app_mode == "Home":
    st.header("Heart Disease Prediction SVM (Support Vector Machine) Model")
    st.text("By Joel Ngiam")
    
    # Add GitHub Repository Link
    st.markdown("[🔗 GitHub Repository](https://github.com/JoelNgiamKeeYong/ml-heart-disease-predictor)", unsafe_allow_html=True)
    
    # Image related to the project
    image_path = "ml-heart-disease-prediction.png"
    st.image(image_path, caption="Heart Disease Prediction Model", width=500)
    
    st.subheader("Context")
    st.text("""
        Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, 
        which accounts for 31% of all deaths worldwide. Four out of 5 CVD deaths are due to heart attacks and strokes, 
        and one-third of these deaths occur prematurely in people under 70 years of age.
        Heart failure is a common event caused by CVDs, and this dataset contains 11 features that can be used to predict heart disease.
        Early detection and management through machine learning models can help at-risk individuals. 
    """)
    
    st.subheader("Attribute Information")
    st.info("1. **Age**: age of the patient [years]")
    st.info("2. **Sex**: sex of the patient [M: Male, F: Female]")
    st.info("3. **ChestPainType**: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]")
    st.info("4. **RestingBP**: resting blood pressure [mm Hg]")
    st.info("5. **Cholesterol**: serum cholesterol [mm/dl]")
    st.info("6. **FastingBS**: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]")
    st.info("7. **RestingECG**: resting electrocardiogram results [Normal: Normal, ST: ST-T wave abnormality, LVH: Left Ventricular Hypertrophy]")
    st.info("8. **MaxHR**: maximum heart rate achieved [Numeric value between 60 and 202]")
    st.info("9. **ExerciseAngina**: exercise-induced angina [Y: Yes, N: No]")
    st.info("10. **Oldpeak**: oldpeak = ST depression [Numeric value]")
    st.info("11. **ST_Slope**: slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]")
    st.info("12. **HeartDisease**: output class [1: heart disease, 0: Normal]")

# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")

    # Mapping dictionaries for user input
    sex_mapping = {"Male": "M", "Female": "F"}
    chest_pain_mapping = {
        "Typical Angina": "TA",
        "Atypical Angina": "ATA",
        "Non-anginal Pain": "NAP",
        "Asymptomatic": "ASY"
    }
    fasting_bp_mapping = {
        "No": 0,
        "Yes": 1
    }
    resting_ecg_mapping = {
        "Normal": "Normal",
        "ST-T wave abnormality": "ST",
        "Left Ventricular Hypertrophy": "LVH"
    }
    exercise_angina_mapping = {"Yes": "Y", "No": "N"}
    st_slope_mapping = {
        "Down": "Down",
        "Flat": "Flat",
        "Up": "Up"
    }

    st.markdown("""
        <style>
            .stTextInput, .stNumberInput, .stSelectbox {
                margin-top: 5px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Collect user inputs
    age = st.number_input("Age", min_value=28, max_value=77, value=40)
    sex = st.selectbox("Sex", list(sex_mapping.keys()))
    chest_pain = st.selectbox("Chest Pain Type", list(chest_pain_mapping.keys()))
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    cholesterol = st.number_input("Cholesterol Level (mm/dl)", min_value=85, max_value=603, value=200)
    fasting_bp = st.selectbox("Fasting Blood Sugar ('Yes' if FastingBS > 120 mg/dl, 'No' if otherwise)", list(fasting_bp_mapping.keys()))
    resting_ecg = st.selectbox("Resting ECG", list(resting_ecg_mapping.keys()))
    max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=202, value=150)
    exercise_angina = st.selectbox("Exercise Induced Angina", list(exercise_angina_mapping.keys()))
    oldpeak = st.number_input("ST Depression", min_value=-2.6, max_value=6.2, value=1.0)
    st_slope = st.selectbox("ST Slope", list(st_slope_mapping.keys()))

    # Map user input to encoded values
    sex = sex_mapping[sex]
    chest_pain = chest_pain_mapping[chest_pain]
    fasting_bp = fasting_bp_mapping[fasting_bp]
    resting_ecg = resting_ecg_mapping[resting_ecg]
    exercise_angina = exercise_angina_mapping[exercise_angina]
    st_slope = st_slope_mapping[st_slope]

    # Button to make prediction
    if st.button("Predict"):
        # Convert categorical values using label encoders
        sex_encoded = le1.transform([sex])[0]
        chest_pain_encoded = le2.transform([chest_pain])[0]
        resting_ecg_encoded = le6.transform([resting_ecg])[0]
        exercise_angina_encoded = le8.transform([exercise_angina])[0]
        st_slope_encoded = le10.transform([st_slope])[0]

        # Prepare input for model (must match feature order used in training)
        input_data = np.array([
            age, sex_encoded, chest_pain_encoded, resting_bp, cholesterol, fasting_bp,
            resting_ecg_encoded, max_hr, exercise_angina_encoded, oldpeak, st_slope_encoded
        ]).reshape(1, -1)

        # Apply the same StandardScaler transformation
        input_data = sc.transform(input_data)  

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Display result with animations using GIFs
        if prediction == 1:
            st.error("The model predicts a high risk of heart disease. Please consult a doctor.")
        else:
            st.success("The model predicts a low risk of heart disease. Stay healthy!")
