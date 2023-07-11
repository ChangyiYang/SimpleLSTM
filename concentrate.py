import os
import pandas as pd

def merge_csv_files(folder_path, output_path, prefix = '' ,files=None):

    files = [ prefix + x  for x in files]
    # Get the paths of all CSV files in the folder
    if files is None:
        files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    print(files)
    
    if not files:
        print("No CSV files found in the folder.")
        return
    
    # Read the first CSV file and create the merged dataframe
    merged_data = pd.read_csv(os.path.join(folder_path, files[0]))
    
    # Loop through the remaining CSV files and merge them into the dataframe
    for file in files[1:]:
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)
        merged_data = pd.concat([merged_data, data], ignore_index=True)
    
    # Save the merged dataframe to the output path
    merged_data.to_csv(output_path, index=False)
    print("CSV files merged successfully! Saved at:", output_path)

# Test

name = 'Xe'
folder_path = f'./Raw Data/Add Xe/{name}'  # Replace with your folder path
output_path = f'./Raw Data/Add Xe/{name}_testing.csv'  # Replace with your output path and filename
prefix = 'Xe135_'

# files = ['P0.6T1', 'P0.8T1', 'P1T0.9', 'P1T1.1']  # Replace with your list of file names
files = ['easy', 'hard']

files = [file + '.csv' for file in files]

merge_csv_files(folder_path, output_path, prefix,files)
