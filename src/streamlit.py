import streamlit as st
import requests

# Header of the UI
st.title("Fire Prediction from IoT Smoke Detector")
# st.subheader("image.png")

# create form of input
with st.form(key="smoke_data_form"):

    # numerics input using box
    temperature_c_ = st.number_input(
        label="1.\tTemperature (C):",
        min_value=-100,
        max_value=1000,
        help="Value range from -100 to 1000 C"
    )

    humidity_percent_ = st.number_input(
        label="2.\tHumidity (%):",
        min_value=0,
        max_value=100,
        help="Value range from 0 to 1000 %"
    )

    tvoc_ppb_ = st.number_input(
        label="3.\tTVOC (ppb):",
        # min_value=-1000
        # max_value=1000
        # help="Value range from -100 to 1000 C"
    )

    eco2_ppm_ = st.number_input(
        label="4.\teCO2 (ppm):",
        # min_value=-1000
        # max_value=1000
        # help="Value range from -100 to 1000 C"
    )

    raw_h2 = st.number_input(
        label="3.\tRaw H2:",
        # min_value=-1000
        # max_value=1000
        # help="Value range from -100 to 1000 C"
    )

    raw_ethanol = st.number_input(
        label="3.\tRaw Ethanol:",
        # min_value=-1000
        # max_value=1000
        # help="Value range from -100 to 1000 C"
    )

    pressure_hpa = st.number_input(
        label="3.\tPressure (HPa):",
        # min_value=-1000
        # max_value=1000
        # help="Value range from -100 to 1000 C"
    )

    pm10 = st.number_input(
        label="3.\tPM10:",
        # min_value=-1000
        # max_value=1000
        # help="Value range from -100 to 1000 C"
    )

    pm25 = st.number_input(
        label="3.\tPM25:",
        # min_value=-1000
        # max_value=1000
        # help="Value range from -100 to 1000 C"
    )

    nc05 = st.number_input(
        label="3.\tNC05:",
        # min_value=-1000
        # max_value=1000
        # help="Value range from -100 to 1000 C"
    )

    nc10 = st.number_input(
        label="3.\tNC10:",
        # min_value=-1000
        # max_value=1000
        # help="Value range from -100 to 1000 C"
    )

    nc25 = st.number_input(
        label="3.\tNC25:",
        # min_value=-1000
        # max_value=1000
        # help="Value range from -100 to 1000 C"
    )

    # button to submit
    submitted = st.form_submit_button("Predict")

    # condition when form submitted
    if submitted:

        # create dict of input data
        raw_data = {
            "temperature_c_": temperature_c_,
            "humidity_percent_": humidity_percent_,
            "tvoc_ppb_": tvoc_ppb_,
            "eco2_ppm_": eco2_ppm_,
            "raw_h2": raw_h2,
            "raw_ethanol": raw_ethanol,
            "pressure_hpa": pressure_hpa,
            "pm10": pm10,
            "pm25": pm25,
            "nc05": nc05,
            "nc10": nc10,
            "nc25": nc25
        }

        # create loading animation while predicting
        with st.spinner("Sending data to prediction server ..."):
            res = requests.post("http://127.0.0.1:8000/predict/", json = raw_data).json()

        # parse the prediction result
        if res["prediction"] == "0":
            st.success("NO FIRE!!!")

        else:
            st.warning("FIRE!!!")