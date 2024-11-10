import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from PIL import Image
import streamlit.components.v1 as components
from glob import glob
import sys

sys.path.append('D:\Python_projects\Anime-Chatbot')
from Character_network import CharacterNetworkGenerator
import numpy as np
# from dotenv import load_dotenv
# load_dotenv()
import plotly.graph_objects as go

files = glob('sample_output.html')

# # Set page title
# st.set_page_config(page_title="Multi-page App", layout="wide")
#
#
# # Home Page Layout
# def home():
#     st.title("Home Page")
#
#     # Top Section: Horizontal Bar Plot on Left, Image on Right
#     top_section = st.container()
#     with top_section:
#         col2 = st.columns(1)
#
#         # Left: Horizontal Bar Plot
#         # with col1:
#         #     st.subheader("Interactive 3D-like Bar Plot")
#         #
#         #     # Sample data for the bar plot
#         #     categories = ['Category A', 'Category B', 'Category C', 'Category D']
#         #     values = [10, 24, 36, 40]
#         #     colors = ['#FF6347', '#4682B4', '#8A2BE2', '#3CB371']  # Custom colors
#         #
#         #     # Create an interactive 3D-like bar plot using Plotly
#         #     fig = go.Figure()
#         #
#         #     # Simulating 3D effect by stacking bars on different 'z' positions
#         #     for i, category in enumerate(categories):
#         #         fig.add_trace(go.Bar(
#         #             x=[category],  # X axis (category)
#         #             y=[values[i]],  # Y axis (values)
#         #             marker=dict(color=colors[i], opacity=0.7),
#         #             hoverinfo='x+y',
#         #             name=category,
#         #             width=0.5  # Bar width
#         #         ))
#         #
#         #     # Update layout for a 3D-like aesthetic
#         #     fig.update_layout(
#         #         barmode='stack',
#         #         xaxis_title="Categories",
#         #         yaxis_title="Values",
#         #         title="Interactive 3D-like Horizontal Bar Plot",
#         #         height=500,
#         #         margin=dict(l=0, r=0, b=0, t=30)
#         #     )
#         #
#         #     # Display interactive Plotly figure in Streamlit
#         #     st.plotly_chart(fig)
#
#         # Right: Image Display
#         with col2:
#             st.subheader("Image")
#             image = Image.open('assets/naruto.jpeg')
#             st.image(image, caption="Sample Image")
#
#     # Bottom Section: Character Network Graph
#     bottom_section = st.container()
#     with bottom_section:
#         st.subheader("Character Network Graph")
#
#         # Create a network graph
#         G = nx.Graph()
#         G.add_edges_from([("Harry", "Ron"), ("Harry", "Hermione"), ("Ron", "Hermione"), ("Harry", "Dumbledore")])
#
#         fig, ax = plt.subplots()
#         pos = nx.spring_layout(G)
#         nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold')
#         st.pyplot(fig)
#
#
# # Navigation Menu
# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to", ["Home", "Page 1", "Page 2", "Page 3"])
#
# # Page Routing
# if page == "Home":
#     home()
# elif page == "Page 1":
#     st.write("This is Page 1")
# elif page == "Page 2":
#     st.write("This is Page 2")
# elif page == "Page 3":
#     st.write("This is Page 3")


# Home.py
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt


# Function to generate a character network graph


def app():
    # st.title("Welcome to the Character Network App")

    #
    # def generate_character_network(character):
    #     G = nx.Graph()
    #     # Example data for character networks; replace with actual data
    #     data = {
    #         'Character A': [('Character A', 'Character B'), ('Character A', 'Character C')],
    #         'Character B': [('Character B', 'Character D'), ('Character B', 'Character E')],
    #         'Character C': [('Character C', 'Character F'), ('Character C', 'Character G')],
    #     }
    #     edges = data.get(character, [])
    #     G.add_edges_from(edges)
    #     return G

    # Streamlit App Code
    st.title("Welcome to the Character Network App")

    # Divide the page into two horizontal sections
    # top_section, bottom_section = st.columns([1, 1])

    # Top section: display an image
    with st.container():
        image = Image.open('assets/naruto.jpeg')
        st.image(image, caption="Character Network Image", use_column_width=True)

    # Bottom section: character network dropdown and graph
    with st.container():
        st.subheader("Select an Episode to view their character network ")

        arr = np.arange(1, 221)
        # print(arr)

        # arr = [str(x) for x in arr]
        episode = st.selectbox("Select Episode", arr, key="Character Network")

        ch_nwtrk = CharacterNetworkGenerator()
        ch_nwtrk.generate_char_network(episode)

        # G = generate_character_network(character)
        #
        # # Plot the network
        # fig, ax = plt.subplots()
        # nx.draw(G, with_labels=True, node_color="skyblue", font_size=12, font_weight="bold", ax=ax)
        # st.pyplot(fig)

        with open(files[0], "r", encoding="utf-8") as html_file:
            html_content = html_file.read()
            components.html(html_content, height=400, scrolling=False)
