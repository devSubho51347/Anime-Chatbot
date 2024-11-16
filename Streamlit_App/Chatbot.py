
import streamlit as st
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# import torch
# from huggingface_hub import login
#
# import pandas as pd
# import torch
# import re
# import huggingface_hub
# from datasets import Dataset
# import transformers
# from transformers import (
#     BitsAndBytesConfig,
#     AutoModelForCausalLM,
#     AutoTokenizer,
# )

from openai import OpenAI
from time import sleep
import os
from dotenv import load_dotenv

# Load environment variables from the secrets file
load_dotenv('D:\Python_projects\Anime-Chatbot\secrets.env')

# Access the API key
openai_api_key = os.getenv('OPENAI_API_KEY')
# print(openai_api_key)

# Initialize OpenAI client
# client = OpenAI(api_key = openai_api_key)
client = OpenAI(api_key= openai_api_key)

job = client.fine_tuning.jobs.retrieve("ftjob-y8L4VubWdpixj580bmko1jzE")
my_model = job.fine_tuned_model


from dummy import load_finetuned_model
from peft import LoraConfig, PeftModel
# from trl import SFTConfig, SFTTrainer

def app():
    # st.title("App Under Development")



    # Replace 'YOUR_TOKEN' with your actual Hugging Face Hub token.
    # You can generate a token at https://huggingface.co/settings/tokens
    # login(token='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')

    def test_model(model_id, test_input):
        """Test the fine-tuned model"""
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": "Your are Naruto from the anime 'Naruto'. Your responses should reflect his personality and speech patterns. All your responses should have a lot of energy and enthusiasm"
                },
                {"role": "user", "content": test_input}
            ]
        )
        return completion.choices[0].message

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Streamlit app layout
    st.title("Chat with Naruto")

    # # Load model directly
    # from transformers import AutoModel
    # model = AutoModel.from_pretrained("devSubho51347/naruto_llama_1b_model_3")

    # model = AutoModelForCausalLM.from_pretrained("devSubho51347/naruto_llama_1b_model_3", torch_dtype=torch.float16,
    #                                              )
    # tokenizer = AutoTokenizer.from_pretrained("devSubho51347/naruto_llama_1b_model_3")
    #
    # # Ensure the tokenizer has the chat template for correct formatting
    # tokenizer.chat_template = "<|system|>{system}\n<|user|>{user}\n<|assistant|>{assistant}"
    #
    # # Set up a pipeline for text generation
    # chat = pipeline("text-generation", model=model, tokenizer=tokenizer)

    local_directory = "../local_model_directory"

    # Load the model and tokenizer from the local directory
    # model = AutoModelForCausalLM.from_pretrained(local_directory)
    # tokenizer = AutoTokenizer.from_pretrained(local_directory,use_fast=False)
    #
    # tokenizer.chat_template = "<|system|>{system}\n<|user|>{user}\n<|assistant|>{assistant}"
    #
    # # Set up a pipeline for text generation
    # chat = pipeline("text-generation", model=model, tokenizer=tokenizer)
    # # chat = load_finetuned_model()

    # System message (optional, you can customize or remove it)
    system_prompt = "Your are Naruto from the anime 'Naruto'. Your responses should reflect his personality and speech patterns"

    st.markdown("""
        <style>
        .chat-container {
            max-width: 600px;
            margin: auto;
            padding: 10px;
            border-radius: 10px;
            background-color: #f0f2f5;
        }
        .user-bubble {
            background-color: #0084FF;
            color: white;
            padding: 10px;
            border-radius: 15px;
            margin: 5px;
            max-width: 70%;
            align-self: flex-end;
        }
        .assistant-bubble {
            background-color: #E5E5EA;
            color: black;
            padding: 10px;
            border-radius: 15px;
            margin: 5px;
            max-width: 70%;
            align-self: flex-start;
        }
        .chat-wrapper {
            display: flex;
            flex-direction: column;
        }
        </style>
    """, unsafe_allow_html=True)

    # User input
    user_input = st.text_input("You:", key="user_input", placeholder="Type your message here...")

    if user_input:
        # Format input using the chat template
        # formatted_input = tokenizer.chat_template.format(system=system_prompt, user=user_input, assistant="")

        # Generate model response
        # response = chat(formatted_input, max_length=100, do_sample=True)
        assistant_response = test_model(my_model,user_input)
        assistant_response = assistant_response.content

        # Append user and assistant messages to session state for display
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.session_state["messages"].append({"role": "assistant", "content": assistant_response})

        # Clear input box after sending a message
        # st.experimental_rerun()

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state["messages"]:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-wrapper"><div class="user-bubble">{message["content"]}</div></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-wrapper"><div class="assistant-bubble">{message["content"]}</div></div>',
                        unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
