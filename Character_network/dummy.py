import pandas as pd

df = pd.read_csv("../naruto_scraper/data/Processed_files/Processed_subtitles.csv")

print(df['Episode'].isnull().sum())