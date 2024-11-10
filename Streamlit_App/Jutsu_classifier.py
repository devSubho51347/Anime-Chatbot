# Jutsu_Classifier.py
import streamlit as st
from glob import glob
import sys

sys.path.append('D:\Python_projects\Anime-Chatbot')
from Jutsu_Classifier_Module import JutsuPredictor


def app():
    st.title("Jutsu Classifier - Finetuned Roberta Transformer")
    st.write("Classify Jutsus based on your input.")

    # Textbox for input
    jutsu_input = st.text_area("Enter the Jutsu or description:", height=100, key="Jutsu Input")

    # Predict button
    if st.button("Predict", key="Jutsu Predictor"):
        # Placeholder for the prediction logic
        # Replace this with your actual model prediction code
        if jutsu_input:

            jutsu = JutsuPredictor(jutsu_input)
            jutsu_type = jutsu.jutsu_output()
            prediction = f"Predicted class for '{jutsu_input}' is: {jutsu_type}"  # Sample output
            st.markdown(f"<span style='color:green; font-weight:bold;'>{prediction}</span>", unsafe_allow_html=True)
        else:
            st.write("Please enter a Jutsu description to get a prediction.")
