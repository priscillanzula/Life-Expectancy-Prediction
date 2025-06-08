import streamlit as st
import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open('random_forest_model.pkl', 'rb'))

st.title("🌍 Life Expectancy Prediction App")

# Define categorical mapping
status_mapping = {"Developing": 1, "Developed": 0}

# User input fields
year = st.number_input("Year", min_value=1900, max_value=2100, value=2023)
adult_mortality = st.number_input("Adult Mortality Rate", min_value=0)
infant_deaths = st.number_input("Infant Deaths", min_value=0)
alcohol = st.number_input("Alcohol Consumption (litres)", min_value=0.0)
percentage_expenditure = st.number_input(
    "Percentage Health Expenditure", min_value=0.0)
hepatitis_b = st.number_input(
    "Hepatitis B (% vaccinated)", min_value=0.0, max_value=100.0)
measles = st.number_input("Measles Cases", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
under_five_deaths = st.number_input("Under-5 Deaths", min_value=0)
polio = st.number_input("Polio (% vaccinated)", min_value=0.0, max_value=100.0)
total_expenditure = st.number_input("Total Health Expenditure", min_value=0.0)
diphtheria = st.number_input(
    "Diphtheria (% vaccinated)", min_value=0.0, max_value=100.0)
hiv_aids = st.number_input("HIV/AIDS Prevalence", min_value=0.0)
gdp = st.number_input("GDP per Capita", min_value=0.0)
population = st.number_input("Population", min_value=0.0)
thinness_1_19 = st.number_input("Thinness 1-19 years (%)", min_value=0.0)
thinness_5_9 = st.number_input("Thinness 5-9 years (%)", min_value=0.0)
income_composition = st.number_input(
    "Income Composition of Resources", min_value=0.0, max_value=1.0)
schooling = st.number_input("Schooling (Years)", min_value=0.0)
status = st.selectbox("Development Status", ["Developing", "Developed"])

# Encode development status as a binary feature
encoded_status = status_mapping[status]

if st.button("Predict"):
    # Create DataFrame matching training features
    input_df = pd.DataFrame([{
        'Year': year,
        'Adult Mortality': adult_mortality,
        'infant deaths': infant_deaths,
        'Alcohol': alcohol,
        'percentage expenditure': percentage_expenditure,
        'Hepatitis B': hepatitis_b,
        'Measles': measles,
        'BMI': bmi,
        'under-five deaths': under_five_deaths,
        'Polio': polio,
        'Total expenditure': total_expenditure,
        'Diphtheria': diphtheria,
        'HIV/AIDS': hiv_aids,
        'GDP': gdp,
        'Population': population,
        'thinness  1-19 years': thinness_1_19,
        'thinness 5-9 years': thinness_5_9,
        'Income composition of resources': income_composition,
        'Schooling': schooling,
        'Status_Developing': encoded_status
    }])

    # Reorder columns to match training set
    expected_columns = ['Year', 'Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure',
                        'Hepatitis B', 'Measles', 'BMI', 'under-five deaths', 'Polio', 'Total expenditure',
                        'Diphtheria', 'HIV/AIDS', 'GDP', 'Population', 'thinness  1-19 years',
                        'thinness 5-9 years', 'Income composition of resources', 'Schooling', 'Status_Developing']

    input_df = input_df[expected_columns]

    prediction = model.predict(input_df)
    st.write(f"Raw model output: {prediction}")
    st.success(f"Predicted Life Expectancy Category: {prediction[0]}")
