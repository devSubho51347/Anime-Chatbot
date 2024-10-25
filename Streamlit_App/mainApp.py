# Main.py
import streamlit as st
from streamlit_option_menu import option_menu
import Character_app
import Jutsu_classifier
import Theme_classifier_app

# Main navigation
# st.title("Multi-Page Streamlit App")

with st.sidebar:
    selected = option_menu(
        "Main Menu",
        ["Home Page", "Jutsu Classifier", "Theme Classifier"],
        # icons=["house", "binoculars", "palette"],
        menu_icon="cast",
        default_index=0,
    )

# Load each page based on sidebar selection
if selected == "Home Page":
    Character_app.app()
elif selected == "Jutsu Classifier":
    Jutsu_classifier.app()
elif selected == "Theme Classifier":
    Theme_classifier_app.app()
