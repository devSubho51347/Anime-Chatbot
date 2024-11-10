# import glob
from glob import glob
import regex as re
import pandas as pd
from datasets import Dataset

import sys

sys.path.append('D:\Python_projects\Anime-Chatbot')

df = pd.read_csv("../naruto_scraper/data/naruto.csv")

print(df.head())

def remove_paranthesis(text):
    result = re.sub(r'\(.*?\)','',text)
    return result


# "prompt": "<|system|>You are a helpful assistant.\n<|user|>What is the capital of France?\n<|assistant|>The capital of France is Paris."

def load_data():
    naruto_transcript_df = pd.read_csv("../naruto_scraper/data/naruto.csv")
    naruto_transcript_df = naruto_transcript_df.dropna()
    naruto_transcript_df['line'] = naruto_transcript_df['line'].apply(remove_paranthesis)
    naruto_transcript_df['number_of_words'] = naruto_transcript_df['line'].str.strip().str.split(" ")
    naruto_transcript_df['number_of_words'] = naruto_transcript_df['number_of_words'].apply(lambda x: len(x))
    naruto_transcript_df['naruto_response_flag'] = 0
    naruto_transcript_df.loc[(naruto_transcript_df['name'] == "Naruto") & (
                naruto_transcript_df['number_of_words'] > 5), 'naruto_response_flag'] = 1

    indexes_to_take = list(naruto_transcript_df[(naruto_transcript_df['naruto_response_flag'] == 1) & (
                naruto_transcript_df.index > 0)].index)

    system_promt = """" <|system|> Your are Naruto from the anime "Naruto". Your responses should reflect his personality and speech patterns \n"""
    prompts = []
    for ind in indexes_to_take:
        prompt = system_promt

        prompt += "<|user|>" +  naruto_transcript_df.iloc[ind - 1]['line']
        prompt += '\n'
        prompt +=  "<|assistant|>" + naruto_transcript_df.iloc[ind]['line']
        prompts.append(prompt)

    df = pd.DataFrame({"prompt": prompts})

    df.to_csv("../naruto_scraper/data/Processed_files/Processed_dialogues.csv")
    print(df.iloc[0])
    dataset = Dataset.from_pandas(df)

    return dataset

new_data = load_data()

# print(new_data.head())