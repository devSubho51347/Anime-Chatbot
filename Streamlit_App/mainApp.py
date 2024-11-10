# Main.py
import streamlit as st
from streamlit_option_menu import option_menu
import Character_app
import Jutsu_classifier
import Theme_classifier_app
import Chatbot

# Main navigation
# st.title("Multi-Page Streamlit App")

with st.sidebar:
    st.set_page_config(layout="wide")

    # Apply custom CSS to remove padding and margins
    st.markdown(
        """
        <style>
        .css-18e3th9 {  /* Removes padding on the main container */
            padding-top: 0;
            padding-bottom: 0;
            padding-left: 0;
            padding-right: 0;
        }
        .css-1d391kg {  /* Removes padding on the sidebar */
            padding-top: 0;
            padding-bottom: 0;
            padding-left: 0;
            padding-right: 0;
        }
        .css-1lcbmhc {  /* Removes padding on top of the container */
            padding-top: 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    selected = option_menu(
        "Main Menu",
        ["Home Page", "Jutsu Classifier", "Theme Classifier", "NaruChatApp"],
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

elif selected == "NaruChatApp":
    Chatbot.app()
