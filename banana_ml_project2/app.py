"""
Streamlit demo: enter a banana's readings, get its estimated remaining shelf life.

    streamlit run app.py

Run `python main.py` first so a trained model exists.
"""

import streamlit as st

import config
from predict import predict_remaining_days

st.set_page_config(page_title="Banana Shelf-Life Predictor", page_icon="🍌")

st.title("🍌 Banana Remaining Shelf-Life Predictor by Bornwise")
st.write(
    "Exploratory machine-learning demo for the final-year project. From today's "
    "ethylene reading and the KMnO4 treatment, it estimates how many more days "
    "the banana has before its shelf life ends (over-ripe / decayed)."
)

day = st.number_input("Storage day", min_value=1, value=3, step=1)
dosage_label = st.selectbox("KMnO4 dosage", ["Control (0 g)", "10 g", "20 g"])
ethylene_ppm = st.number_input("Ethylene concentration today (ppm)", min_value=0.0, value=50.0)

colour_value = None
if config.USE_COLOUR_AS_FEATURE:
    colour_value = st.slider("Peel colour stage (1-7)", 1, 7, 3)

dosage_map = {"Control (0 g)": "None", "10 g": "10 g", "20 g": "20 g"}

if st.button("Estimate remaining shelf life"):
    try:
        est, low, high = predict_remaining_days(
            day, dosage_map[dosage_label], ethylene_ppm, colour_value=colour_value)
        st.success(f"Estimated shelf life left: **about {est:.1f} days**")
        st.write(f"Likely range: {low:.1f} to {high:.1f} days")
        st.caption("Trained on a small laboratory dataset - a proof-of-concept estimate, "
                   "not a guarantee.")
    except FileNotFoundError:
        st.error("No trained model found. Run `python main.py` first.")
