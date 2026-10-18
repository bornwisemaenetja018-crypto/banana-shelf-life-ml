"""
Streamlit demo app for the banana KMnO4 shelf-life project.

Loads the already-trained model (via predict.py) and lets you enter one
banana's measurements to get a live quality-stage prediction — for the
poster session / oral presentation demo.

Run with:
    streamlit run app.py

Requires a trained model to already exist (run `python main.py` first).
"""

import streamlit as st

from predict import predict_quality

st.set_page_config(page_title="Banana Quality Predictor", page_icon="🍌")

st.title("Banana Quality Prediction Model")
st.write(
    "Exploratory machine-learning interface for the final-year project: "
    "predicts banana quality stage from ethylene concentration, peel colour "
    "stage, storage conditions and KMnO4 dosage, using a Random Forest model "
    "trained on the potassium permanganate storage trial by Bornwise K. Maenetja."
)

day = st.number_input("Storage day", min_value=0, value=1, step=1)
dosage_label = st.selectbox("KMnO4 dosage", ["Control (0 g)", "10 g", "20 g"])
ethylene_ppm = st.number_input("Ethylene concentration (ppm)", min_value=0.0, value=0.0)
relative_humidity = st.number_input("Relative humidity (%)", min_value=0.0, max_value=100.0, value=0.40)
storage_temperature = st.number_input("Storage temperature (°C)", value=25.0)
colour_value = st.slider("Peel colour stage (1 = green, 7 = brown-flecked)", min_value=1, max_value=7, value=1)

dosage_map = {"Control (0 g)": "0 g", "10 g": "10 g", "20 g": "20 g"}

if st.button("Predict quality"):
    try:
        prediction, probabilities = predict_quality(
            day,
            dosage_map[dosage_label],
            ethylene_ppm,
            relative_humidity,
            storage_temperature,
            colour_value,
        )

        st.success(f"Predicted quality stage: **{prediction}**")

        st.subheader("Model probabilities")
        for label, probability in sorted(probabilities.items(), key=lambda x: -x[1]):
            st.write(f"{label}: {probability:.1%}")
            st.progress(float(probability))

    except FileNotFoundError:
        st.error(
            "No trained model found. Run `python main.py` first to train "
            "and save a model before using this app."
        )
