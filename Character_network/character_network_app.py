### Create 3 methods, one for dataloader, one for NER model and the last one for character network

from transformers import pipeline
from nltk import sent_tokenize
import nltk
import torch
from glob import glob
import regex as re
import pandas as pd
import os



# nltk.download('punkt')

## get all the subtitle files

files = glob('../naruto_scraper/data/Subtitles/*')

directory = '../naruto_scraper/data/Subtitles/'
# print(files[:10])

subtitles_dict = {}

# print(files[15:19])

new_files = []
for ele in files:
    new_files.append(directory + ele.split("\\")[-1])
print(len(new_files))

# # print(new_files[25])
#
# for ele in new_files[25:40]:
#     print(ele)
#     with open(ele, 'r', encoding='utf-8', errors='ignore') as file:
#         line = file.readlines()
#         # print(line)


data_dict = {}
data_dict['Subtitles'] = []
data_dict['Episode'] = []

for ele in new_files:
    try:
        with open(ele,'r',encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()
            lines = lines[27:]
            lines = [",".join(line.split(",")[9:]) for line in lines]
            lines = [line.replace('\\N',' ') for line in lines]

            # lines = [". ".join(line) for line in lines]
            lines = " ".join(lines)
            data_dict['Subtitles'].append(lines)
            # print(lines)

    ## Get the episode number using regex

        episode_num = re.search(r'-\s*(\d+)\s*\.(ass|srt)', ele).group(1)
        data_dict['Episode'].append(episode_num)
        print(episode_num)

    except:
        pass
#
# print(data_dict)
#
df = pd.DataFrame(data = data_dict)
df.to_csv("../naruto_scraper/data/Processed_files/Processed_subtitles.csv", index = False)
