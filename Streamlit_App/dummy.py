from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from huggingface_hub import hf_hub_download
#
# # Clear cache for fresh download
# # hf_hub_download.clear_cache()
#
# # Load model again
#
def load_finetuned_model():

    model = AutoModelForCausalLM.from_pretrained("devSubho51347/naruto_llama_1b_model_3")
    tokenizer = AutoTokenizer.from_pretrained("devSubho51347/naruto_llama_1b_model_3")

# Ensure the tokenizer has the chat template for correct formatting
    tokenizer.chat_template = "<|system|>{system}\n<|user|>{user}\n<|assistant|>{assistant}"

# Set up a pipeline for text generation
    chat = pipeline("text-generation", model=model, tokenizer=tokenizer)

    return chat
#
# # system_prompt = "You are a helpful assistant."
# # formatted_input = tokenizer.chat_template.format(system=system_prompt, user="What is your name ", assistant="")
# #
# #         # Generate model response
# # response = chat(formatted_input, max_length=100, do_sample=True)
# # assistant_response = response[0]["generated_text"].split("<|assistant|>")[-1].strip()
# #
# # print(assistant_response)a


# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# # Specify the model repository ID on Hugging Face
# model_name_or_path = "devSubho51347/naruto_llama_1b_model_3"
#
# # Load the model and tokenizer from the Hugging Face Hub
# model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
#
# # Define a local directory to save the model and tokenizer
# save_directory = "../local_model_directory"
#
# # Save the model and tokenizer to the local directory
# model.save_pretrained(save_directory)
# tokenizer.save_pretrained(save_directory)
#
# print(f"Model and tokenizer saved locally in {save_directory}")
