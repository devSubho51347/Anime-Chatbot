import pandas as pd
from transformers import pipeline
import nltk
import torch

# Use a pipeline as a high-level helper
from transformers import pipeline
from glob import glob
files = glob('../naruto_scraper/data/Subtitles/*')

pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

val = pipe(
    "I have a problem with my iphone that needs to be resolved asap!!",
    candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
    multi_label = True
)

print(val)






df = pd.read_csv("../naruto_scraper/data/Processed_files/Processed_subtitles.csv")