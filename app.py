import streamlit as st
import pandas as pd
import joblib

# Load trained model (based on Top 5 selected features)
model = joblib.load("random_forest_life_expectancy.pkl")

st.title("🌍 Life Expectancy Prediction App")

# Collect input data for the 5 selected features
adult_mortality = st.number_input("Adult Mortality Rate", min_value=0)
bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0)
hiv_aids = st.number_input("HIV/AIDS Prevalence Rate", min_value=0.0)
income_comp = st.slider("Income Composition of Resources (0-1)", 0.0, 1.0, 0.5)
schooling = st.number_input("Average Years of Schooling", min_value=0.0)

# Prepare input DataFrame
input_df = pd.DataFrame([{
    'Adult Mortality': adult_mortality,
    'BMI': bmi,
    'HIV/AIDS': hiv_aids,
    'Income composition of resources': income_comp,
    'Schooling': schooling
}])

# Prediction button
if st.button("Predict Life Expectancy"):
    prediction = model.predict(input_df)[0]
    st.success(f"🎯 Predicted Life Expectancy: {prediction:.2f} years")
