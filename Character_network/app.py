import gradio as gr
import os


# Define a function to read and return the contents of the HTML file
def render_html_file():
    # Path to the HTML file (you can generate this dynamically)
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
    inputs=[],  # No inputs needed
    outputs=gr.HTML(),  # Output component to display HTML
    title="Network Visualization",  # Title of the Gradio app
    description="This app renders an HTML file with network visualization."
)

# Launch the Gradio app
app.launch()
