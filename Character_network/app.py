import gradio as gr
import os
from ner_extractor import generate_char_network as gcn


# Define a function to read and return the contents of the HTML file
def render_html_file(episode):
    # Path to the HTML file (you can generate this dynamically)

    episode = int(episode)
    gcn(episode)

    html_file_path = "sample_output.html"

    # Check if the file exists
    if os.path.exists(html_file_path):
        with open(html_file_path, "r", encoding="utf-8") as file:
            html_content = file.read()
        return html_content
    else:
        return "<h1>Error: HTML file not found!</h1>"


# Create the Gradio interface
app = gr.Interface(
    fn=render_html_file,  # Function to render the HTML
    inputs=gr.Dropdown(choices=['20', '22', '23', '24', '25'], label="Select a Value"),   # No inputs needed
    outputs=gr.HTML(),  # Output component to display HTML
    title="Network Visualization",  # Title of the Gradio app
    description="This app renders an HTML file with network visualization."
)

# Launch the Gradio app
app.launch()
