import pandas as pd
from transformers import pipeline
import nltk
import torch

# Use a pipeline as a high-level helper
from transformers import pipeline
from glob import glob
files = glob('../naruto_scraper/data/Subtitles/*')

pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

df = pd.read_csv("../naruto_scraper/data/Processed_files/Processed_subtitles.csv")

script = df['Subtitles'][0]

val = pipe(
    script,
    candidate_labels=["dialogue", "betrayal", "fight", "respect", "personal development","love","friendship","sacrifice"],
    multi_label = True
)

print(val)






