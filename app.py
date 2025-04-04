import streamlit as st
import pickle
import numpy as np

import warnings
warnings.filterwarnings('ignore')

def load_model(disease):
    if disease == "Cancer":
        return pickle.load(open("cancer_model.pkl", "rb"))
    elif disease == "Diabetes":
        return pickle.load(open("diabetes_model.pkl", "rb"))
    elif disease == "Heart Attack":
        return pickle.load(open("heart_model.pkl", "rb"))
    return None

def main():
    st.title("Multiple Disease Prediction App")
    disease = st.selectbox("Select Disease", ["Cancer", "Diabetes", "Heart Attack"])
    
    if disease:
        model = load_model(disease)
        if model:
            st.subheader(f"Enter details for {disease} Prediction")
            
            if disease == "Cancer":
                age = st.number_input("Age", min_value=1, max_value=120, value=30)
                gender = st.selectbox("Gender", ["Male", "Female"])
                bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
                smoking = st.selectbox("Smoking", ["No", "Yes"])
                genetic_risk = st.number_input("Genetic Risk", min_value=0, max_value=5, value=1)
                physical_activity = st.number_input("Physical Activity", min_value=0.0, max_value=10.0, value=5.0)
                alcohol_intake = st.number_input("Alcohol Intake", min_value=0.0, max_value=10.0, value=2.0)
                cancer_history = st.selectbox("Cancer History", ["No", "Yes"])
                
                gender = 1 if gender == "Female" else 0
                smoking = 1 if smoking == "Yes" else 0
                cancer_history = 1 if cancer_history == "Yes" else 0
                input_data = np.array([[age, gender, bmi, smoking, genetic_risk, physical_activity, alcohol_intake, cancer_history]])
            
            elif disease == "Diabetes":
                pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
                glucose = st.number_input("Glucose Level", min_value=50, max_value=300, value=100)
                blood_pressure = st.number_input("Blood Pressure", min_value=30, max_value=200, value=80)
                skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
                insulin = st.number_input("Insulin", min_value=0, max_value=500, value=100)
                bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
                diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
                age = st.number_input("Age", min_value=1, max_value=120, value=30)
                
                input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
            
            elif disease == "Heart Attack":
                age = st.number_input("Age", min_value=1, max_value=120, value=30)
                gender = st.selectbox("Gender", ["Male", "Female"])
                cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=500, value=200)
                bp = st.number_input("Blood Pressure", min_value=50, max_value=250, value=120)
                heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=75)
                obesity = st.selectbox("Obesity", ["No", "Yes"])
                smoking = st.selectbox("Smoking", ["No", "Yes"])
                
                gender = 1 if gender == "Female" else 0
                obesity = 1 if obesity == "Yes" else 0
                smoking = 1 if smoking == "Yes" else 0
                input_data = np.array([[age, gender, cholesterol, bp, heart_rate, obesity, smoking]])
                
            if st.button("Predict"):
                prediction = model.predict(input_data)[0]
                result = "Positive" if prediction == 1 else "Negative"
                st.success(f"Prediction: {result}")
        else:
            st.error("Model not found!")

if __name__ == "__main__":
    main()
