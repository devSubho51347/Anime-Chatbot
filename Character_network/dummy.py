import pandas as pd
import numpy as np

arr = np.arange(1, 221)
# print(arr)
#
# arr = [str(x) for x in arr]
# print(arr)

df = pd.read_csv("../naruto_scraper/data/Processed_files/Processed_subtitles.csv")

text = df['Subtitles'][80]
print(text)