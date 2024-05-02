import os
import pandas as pd


csv_path = "E:/Data Science/Machine Learning projects/Lahore15Analysis/PoliceStations" # Path to the directory containing CSV files


excel_file = "Data/Crime Calls Data.xlsx"


df = pd.read_excel(excel_file)
ps = df['ps_station'].unique()

for ps_name in ps:
    new_df = df[df['ps_station'] == ps_name]
    new_df['category'] = new_df['category'].str.replace('/', '_')
    categories = new_df['category'].unique() 
    
    ps_folder = os.path.join(csv_path, ps_name)     # Create a folder for each police station if it doesn't exist
    if not os.path.exists(ps_folder):
        os.mkdir(ps_folder)
    

    for category in categories:     # Loop through each category
        category_df = new_df[new_df['category'] == category]        
        category_df.to_csv(f"{ps_folder}/{category}.csv", index=False)
