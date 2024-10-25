import streamlit as st
import sys

sys.path.append('D:\Python_projects\Anime-Chatbot')
from matplotlib import pyplot as plt
from Theme_Classifier import ThemeClassifier
import numpy as np


def app():
    st.title("Classify Theme of the Episode - Zero Shot Classification")

    with st.container():
        # st.subheader("Select an Episode to view their character network ")

        arr = np.arange(1, 221)
        # print(arr)

        # arr = [str(x) for x in arr]
        new_episode = st.selectbox("Select Episode", arr, key="Theme_selector")

        theme = ThemeClassifier(new_episode)
        get_theme = theme.predictTheme()

        if st.button("Classify", key="Theme Predictor"):
            # Placeholder data for bar plot (replace with actual classification data)
            categories = get_theme['labels']
            values = get_theme['scores']  # Sample values

            # Plotting the bar plot
            fig, ax = plt.subplots()
            colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
            ax.bar(categories, values, color=colors)
            ax.set_title(f"Classification Results for Episode No {new_episode}")
            ax.set_xlabel("Categories")
            plt.xticks(rotation=45, ha="right")
            ax.set_ylabel("Values")

            # Display the plot
            st.pyplot(fig)
