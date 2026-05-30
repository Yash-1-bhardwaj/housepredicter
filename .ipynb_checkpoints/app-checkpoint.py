import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("simple_house_price_model.pkl")

st.title("House Price Predictor")

st.write("Enter house details below")

# User Inputs
overall_qual = st.slider("Overall Quality", 1, 10, 5)

gr_liv_area = st.number_input("Ground Living Area", 500, 10000, 1500)

garage_cars = st.slider("Garage Capacity", 0, 5, 2)

total_bsmt_sf = st.number_input("Basement Area", 0, 5000, 800)

first_flr_sf = st.number_input("First Floor Area", 500, 5000, 1000)

second_flr_sf = st.number_input("Second Floor Area", 0, 5000, 500)

full_bath = st.slider("Full Bathrooms", 0, 5, 2)

half_bath = st.slider("Half Bathrooms", 0, 3, 1)

bsmt_full_bath = st.slider("Basement Full Bath", 0, 3, 0)

bsmt_half_bath = st.slider("Basement Half Bath", 0, 2, 0)

year_built = st.number_input("Year Built", 1900, 2025, 2000)

year_remod = st.number_input("Year Remodeled", 1900, 2025, 2005)

yr_sold = st.number_input("Year Sold", 2006, 2010, 2010)

garage_area = st.number_input("Garage Area", 0, 2000, 500)

pool_area = st.number_input("Pool Area", 0, 2000, 0)

# Feature Engineering
total_sf = total_bsmt_sf + first_flr_sf + second_flr_sf

total_bathrooms = (
    full_bath +
    (0.5 * half_bath) +
    bsmt_full_bath +
    (0.5 * bsmt_half_bath)
)

house_age = yr_sold - year_built

remodel_age = yr_sold - year_remod

has_garage = 0 if garage_area == 0 else 1

has_pool = 0 if pool_area == 0 else 1

# Create dataframe
input_data = pd.DataFrame({
    "OverallQual": [overall_qual],
    "GrLivArea": [gr_liv_area],
    "GarageCars": [garage_cars],
    "TotalBsmtSF": [total_bsmt_sf],
    "1stFlrSF": [first_flr_sf],
    "2ndFlrSF": [second_flr_sf],
    "FullBath": [full_bath],
    "HalfBath": [half_bath],
    "BsmtFullBath": [bsmt_full_bath],
    "BsmtHalfBath": [bsmt_half_bath],
    "YearBuilt": [year_built],
    "YearRemodAdd": [year_remod],
    "YrSold": [yr_sold],
    "GarageArea": [garage_area],
    "PoolArea": [pool_area],
    "TotalSF": [total_sf],
    "TotalBathrooms": [total_bathrooms],
    "HouseAge": [house_age],
    "RemodelAge": [remodel_age],
    "HasGarage": [has_garage],
    "HasPool": [has_pool]
})

# Predict
if st.button("Predict Price"):

    prediction_log = model.predict(input_data)

    prediction = np.expm1(prediction_log)

    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")