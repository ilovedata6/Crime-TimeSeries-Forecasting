import pandas as pd
import numpy as np
import os

print("Processing Started...")

df = pd.read_excel("Data/Crime Calls Data.xlsx")


df['category'] = df['category'].str.replace('/', '_')
categories = df['category'].unique()


for i in categories:
    new_df = df[df['category'] == i]
    if not os.path.exists("E:/Data Science/Machine Learning projects/Lahore15Analysis/Categories"):
        os.mkdir("E:/Data Science/Machine Learning projects/Lahore15Analysis/Categories")
    new_df.to_csv(f"./Categories/{i}.csv")
print("Processing Finished")