import requests
import streamlit as st

import os
API_URL = os.getenv("API_URL", "http://localhost:8000") + "/predict"

DATE_OPTIONS = {
    "August 2023": 6,
    "September 2023": 7,
    "October 2023": 8,
    "November 2023": 9,
    "December 2023": 10,
    "January 2024": 0,
    "February 2024": 1,
    "March 2024": 2,
    "April 2024": 3,
    "May 2024": 4,
    "June 2024": 5,
}

st.title("Warsaw Apartment Price Predictor")
st.markdown("Enter apartment details below to get a price estimate.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Basic Info")
    square_meters = st.number_input("Size (sqm)", min_value=10.0, max_value=300.0, value=55.0, step=1.0)
    floor = st.number_input("Floor", min_value=0, max_value=50, value=3, step=1)
    floor_count = st.number_input("Total floors in building", min_value=1, max_value=50, value=5, step=1)
    build_year = st.number_input("Build year", min_value=1850, max_value=2024, value=2005, step=1)
    date_label = st.selectbox("Listing month", options=list(DATE_OPTIONS.keys()), index=5)

with col2:
    st.subheader("Location")
    latitude = st.number_input("Latitude", min_value=52.10, max_value=52.37, value=52.23, format="%.5f")
    longitude = st.number_input("Longitude", min_value=20.82, max_value=21.25, value=21.01, format="%.5f")
    centre_distance = st.number_input("Distance to city centre (km)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
    poi_count = st.number_input("Points of interest nearby", min_value=0, max_value=150, value=20, step=1)

st.subheader("Distances to amenities (km)")
col3, col4 = st.columns(2)

with col3:
    school_distance = st.number_input("School", min_value=0.0, max_value=5.0, value=0.3, step=0.05)
    clinic_distance = st.number_input("Clinic", min_value=0.0, max_value=5.0, value=0.5, step=0.05)
    post_office_distance = st.number_input("Post office", min_value=0.0, max_value=5.0, value=0.4, step=0.05)
    kindergarten_distance = st.number_input("Kindergarten", min_value=0.0, max_value=5.0, value=0.2, step=0.05)

with col4:
    restaurant_distance = st.number_input("Restaurant", min_value=0.0, max_value=5.0, value=0.2, step=0.05)
    college_distance = st.number_input("College", min_value=0.0, max_value=5.0, value=1.5, step=0.05)
    pharmacy_distance = st.number_input("Pharmacy", min_value=0.0, max_value=5.0, value=0.3, step=0.05)

st.divider()

if st.button("Predict Price", type="primary", use_container_width=True):
    payload = {
        "squareMeters": square_meters,
        "floor": floor,
        "floorCount": floor_count,
        "buildYear": build_year,
        "latitude": latitude,
        "longitude": longitude,
        "centreDistance": centre_distance,
        "poiCount": poi_count,
        "schoolDistance": school_distance,
        "clinicDistance": clinic_distance,
        "postOfficeDistance": post_office_distance,
        "kindergartenDistance": kindergarten_distance,
        "restaurantDistance": restaurant_distance,
        "collegeDistance": college_distance,
        "pharmacyDistance": pharmacy_distance,
        "date": DATE_OPTIONS[date_label],
    }

    with st.spinner("Predicting..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=5)
            response.raise_for_status()
            result = response.json()

            st.success("Prediction complete")
            st.metric(
                label="Estimated Price",
                value=result["predicted_price_formatted"]
            )
            price_per_sqm = result["predicted_price"] / square_meters
            st.caption(f"≈ {price_per_sqm:,.0f} PLN/sqm")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API. Make sure `uvicorn main:app --reload` is running.")
        except Exception as e:
            st.error(f"Error: {e}")
