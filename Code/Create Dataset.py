#!/usr/bin/env python
# coding: utf-8

# # ABS of CIR

# In[10]:


import os
import csv
import math

# Function to process CSV files in a folder
def process_csv_files(folder_path):
    output_folder = "C:\\Users\\aligh\\OneDrive\\Desktop\\dd2"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                relative_path = os.path.relpath(root, folder_path)
                output_subfolder = os.path.join(output_folder, relative_path)
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)

                file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_subfolder, file)
                
                with open(file_path, 'r') as csv_file, open(output_file_path, 'w', newline='') as output_file:
                    csv_reader = csv.reader(csv_file)
                    csv_writer = csv.writer(output_file)
                    
                    for row in csv_reader:
                        processed_row = []
                        for i, column in enumerate(row):
                            if i >= 16:  # Assuming column numbering starts from 0 and we process from column 17 onwards
                                try:
                                    complex_num = complex(column)
                                    abs_value = abs(complex_num)
                                    rounded_abs_value = int(round(abs_value))
                                    processed_row.append(rounded_abs_value)
                                except ValueError:
                                    processed_row.append(column)
                            else:
                                processed_row.append(column)
                        
                        csv_writer.writerow(processed_row)

# Replace 'C:\\Users\\aligh\\OneDrive\\Desktop\\d1' with your folder path containing CSV files
process_csv_files("C:\\Users\\aligh\\OneDrive\\Desktop\\d1")


# # Remove other columns

# In[18]:


import os
import csv

# Function to process CSV files in a folder and delete specified columns
def process_csv_files(folder_path):
    output_folder = "C:\\Users\\aligh\\OneDrive\\Desktop\\dd3"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                relative_path = os.path.relpath(root, folder_path)
                output_subfolder = os.path.join(output_folder, relative_path)
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)

                file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_subfolder, file)
                
                with open(file_path, 'r') as csv_file, open(output_file_path, 'w', newline='') as output_file:
                    csv_reader = csv.reader(csv_file)
                    csv_writer = csv.writer(output_file)
                    
                    for row in csv_reader:
                        # Delete specified columns indices
                        modified_row = [val for idx, val in enumerate(row) if idx + 1 not in [1, 2, 6, 7, 8, 11, 13, 14, 15, 16, 18, 20, 21, 22, 23, 24, 25, 26]]
                        csv_writer.writerow(modified_row)

# Replace 'C:\\Users\\aligh\\OneDrive\\Desktop\\dd2' with your folder path containing CSV files
process_csv_files("C:\\Users\\aligh\\OneDrive\\Desktop\\dd2")


# # 1_8

# In[25]:


import os
import csv

# Function to combine CSV files in a folder in groups and save the combined files
def combine_csv_files(folder_path):
    output_folder = "C:\\Users\\aligh\\OneDrive\\Desktop\\dd4"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(folder_path):
        csv_files = [file for file in files if file.endswith(".csv")]
        num_files = len(csv_files)
        if num_files > 0:
            num_iterations = num_files // 8 + (1 if num_files % 8 > 0 else 0)

            for i in range(num_iterations):
                start_idx = i * 8
                end_idx = min((i + 1) * 8, num_files)
                output_subfolder = os.path.join(output_folder, os.path.relpath(root, folder_path))
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)

                combined_file_path = os.path.join(output_subfolder, f"{start_idx + 1}_{end_idx}.csv")

                combined_data = []
                num_rows = 0

                for j in range(start_idx, end_idx):
                    csv_file_path = os.path.join(root, csv_files[j])
                    with open(csv_file_path, 'r') as csv_file:
                        csv_reader = csv.reader(csv_file)
                        csv_data = list(csv_reader)

                        if j == start_idx:
                            combined_data = csv_data.copy()
                            num_rows = len(combined_data)
                        else:
                            for row_idx, row in enumerate(csv_data):
                                if row_idx < num_rows:
                                    combined_data[row_idx].extend(row)

                with open(combined_file_path, 'w', newline='') as combined_file:
                    csv_writer = csv.writer(combined_file)
                    csv_writer.writerows(combined_data)

# Replace 'C:\\Users\\aligh\\OneDrive\\Desktop\\dd3' with your folder path containing CSV files
combine_csv_files("C:\\Users\\aligh\\OneDrive\\Desktop\\dd3")


# In[30]:


import os
import csv

# Function to process CSV files in a folder and update column names from 9th column onwards
def process_csv_files(folder_path):
    output_folder = "C:\\Users\\aligh\\OneDrive\\Desktop\\dd4"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                relative_path = os.path.relpath(root, folder_path)
                output_subfolder = os.path.join(output_folder, relative_path)
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)

                file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_subfolder, file)
                
                with open(file_path, 'r') as csv_file, open(output_file_path, 'w', newline='') as output_file:
                    csv_reader = csv.reader(csv_file)
                    csv_writer = csv.writer(output_file)
                    header_written = False
                    
                    for row in csv_reader:
                        if not header_written:
                            header_written = True
                            updated_header_row = row[:10] + ['CIR' + str(i) for i in range(1, 1016)]
                            csv_writer.writerow(updated_header_row)
                        else:
                            csv_writer.writerow(row)

# Replace 'C:\\Users\\aligh\\OneDrive\\Desktop\\dd3' with your folder path containing CSV files
process_csv_files("C:\\Users\\aligh\\OneDrive\\Desktop\\dd3")


# In[41]:


import os
import csv

# Function to combine the first 8 CSV files into one file, the next 8 into another file, and so on
def combine_csv_files(folder_path):
    output_folder = "C:\\Users\\aligh\\OneDrive\\Desktop\\dd5"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(folder_path):
        file_count = len(files)
        csv_files = [f for f in files if f.endswith(".csv")]

        for i in range(0, file_count, 8):
            file_chunk = csv_files[i:i + 8]
            if not file_chunk:
                break

            data_to_write = []
            header_written = False

            output_file_name = f"{i + 1}_{i + 8}.csv"
            output_subfolder = os.path.join(output_folder, os.path.relpath(root, folder_path))
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)

            output_file_path = os.path.join(output_subfolder, output_file_name)

            for file in file_chunk:
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    data = list(csv_reader)
                    if not header_written:
                        data_to_write.extend(data)
                        header_written = True
                    else:
                        data_to_write = [x + y[8:] for x, y in zip(data_to_write, data)]

            with open(output_file_path, 'w', newline='') as output_file:
                csv_writer = csv.writer(output_file)
                csv_writer.writerows(data_to_write)

# Replace 'C:\\Users\\aligh\\OneDrive\\Desktop\\dd4' with your folder path containing CSV files
combine_csv_files("C:\\Users\\aligh\\OneDrive\\Desktop\\dd4")


# # Stack in column

# In[45]:


import os
import csv

# Function to combine the first 8 CSV files into one file, the next 8 into another file, and so on
def combine_csv_files(folder_path):
    output_folder = "C:\\Users\\aligh\\OneDrive\\Desktop\\dd5"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(folder_path):
        file_count = len(files)
        csv_files = [f for f in files if f.endswith(".csv")]

        for i in range(0, file_count, 8):
            file_chunk = csv_files[i:i + 8]
            if not file_chunk:
                break

            data_to_write = []
            header_written = False

            output_file_name = f"{i + 1}_{i + 8}.csv"
            output_subfolder = os.path.join(output_folder, os.path.relpath(root, folder_path))
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)

            output_file_path = os.path.join(output_subfolder, output_file_name)

            for file in file_chunk:
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    data = list(csv_reader)
                    if not header_written:
                        data_to_write.extend(data)
                        header_written = True
                    else:
                        for idx, row in enumerate(data_to_write):
                            # Append columns from the current CSV file to the existing data
                            data_to_write[idx] += data[idx]

            with open(output_file_path, 'w', newline='') as output_file:
                csv_writer = csv.writer(output_file)
                csv_writer.writerows(data_to_write)

# Replace 'C:\\Users\\aligh\\OneDrive\\Desktop\\dd4' with your folder path containing CSV files
combine_csv_files("C:\\Users\\aligh\\OneDrive\\Desktop\\dd4")


# # Stack in row

# In[16]:


import os
import csv

# Function to stack CSV files in each folder row-wise while removing headers
def stack_csv_files(folder_path):
    output_folder = "C:\\Users\\aligh\\OneDrive\\Desktop\\111"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(folder_path):
        csv_files = [f for f in files if f.endswith(".csv")]
        
        if len(csv_files) > 0:
            folder_name = os.path.relpath(root, folder_path)
            output_subfolder = os.path.join(output_folder, folder_name)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)

            output_file_path = os.path.join(output_subfolder, "stacked.csv")

            data_to_write = []
            header_written = False

            for file in csv_files:
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    data = list(csv_reader)
                    if not header_written:
                        data_to_write.extend(data)
                        header_written = True
                    else:
                        data_to_write.extend(data[1:])

            with open(output_file_path, 'w', newline='') as output_file:
                csv_writer = csv.writer(output_file)
                csv_writer.writerows(data_to_write)

# Replace 'C:\\Users\\aligh\\OneDrive\\Desktop\\dd5' with your folder path containing CSV files
stack_csv_files("C:\\Users\\aligh\\OneDrive\\Desktop\\1.2_6.51_1.32")


# # All part

# In[4]:


import os
import csv

# Function to combine CSV files from each folder into one file row-wise
def combine_csv_files(folder_path):
    output_folder = "C:\\Users\\aligh\\OneDrive\\Desktop\\dd8"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_data = []

    folder_list = sorted(os.listdir(folder_path))  # Get folders sorted alphabetically

    for folder_name in folder_list:
        folder_dir = os.path.join(folder_path, folder_name)

        if os.path.isdir(folder_dir):
            csv_files = [f for f in os.listdir(folder_dir) if f.endswith(".csv")]

            for file in csv_files:
                file_path = os.path.join(folder_dir, file)
                with open(file_path, 'r') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    data = list(csv_reader)
                    all_data.extend(data)

    output_file_path = os.path.join(output_folder, "all_part.csv")
    with open(output_file_path, 'w', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerows(all_data)

# Replace 'C:\\Users\\aligh\\OneDrive\\Desktop\\dd7' with your folder path containing folders with CSV files
combine_csv_files("C:\\Users\\aligh\\OneDrive\\Desktop\\dd7")


# # first row

# In[17]:


import os
import csv
import shutil

# Function to remove the first row from each CSV file in each folder
def remove_first_row(folder_path):
    output_folder = "C:\\Users\\aligh\\OneDrive\\Desktop\\dd7"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                input_file_path = os.path.join(root, file)
                output_subfolder = os.path.join(output_folder, os.path.relpath(root, folder_path))

                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)

                output_file_path = os.path.join(output_subfolder, file)

                with open(input_file_path, 'r') as csv_file, open(output_file_path, 'w', newline='') as output_file:
                    csv_reader = csv.reader(csv_file)
                    csv_writer = csv.writer(output_file)

                    next(csv_reader)  # Skip the first row

                    for row in csv_reader:
                        csv_writer.writerow(row)

# Replace 'C:\\Users\\aligh\\OneDrive\\Desktop\\dd6' with your folder path containing folders with CSV files
remove_first_row("C:\\Users\\aligh\\OneDrive\\Desktop\\dd6")


# # Rename

# In[15]:


import os

# Function to rename CSV files and save in a single folder
def rename_and_save_csv_files(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_count = 0

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_count += 1
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_folder, f"{file_count}.csv")

                # Rename and move the file to the output folder
                os.rename(input_file_path, output_file_path)

# Replace 'C:\\Users\\aligh\\OneDrive\\Desktop\\dd7' with your folder path containing folders with CSV files
# Replace 'C:\\Users\\aligh\\OneDrive\\Desktop\\dd8' with the desired output folder path
rename_and_save_csv_files("C:\\Users\\aligh\\OneDrive\\Desktop\\dd7", "C:\\Users\\aligh\\OneDrive\\Desktop\\dd8")


# In[24]:


import os

# Custom key function to extract numerical value from file names for sorting
def custom_key(folder_name):
    # Extract the numerical part of the folder name
    return tuple(map(float, folder_name.split('_')[0].split('.')))  # Assumes the format is 'x.y_z'

# Function to display folder list sorted based on numerical values in file names
def display_folder_list(folder_path):
    folder_list = sorted([f.name for f in os.scandir(folder_path) if f.is_dir()], key=custom_key)

    print("Folders in the directory:")
    for folder_name in folder_list:
        print(folder_name)

# Replace 'C:\\Users\\aligh\\OneDrive\\Desktop\\dd7' with your folder path containing folders
display_folder_list("C:\\Users\\aligh\\OneDrive\\Desktop\\dd7")


# In[25]:


import os
import csv

# Custom key function to extract numerical value from folder names for sorting
def custom_key(folder_name):
    return tuple(map(float, folder_name.split('_')[0].split('.')))  # Extract numerical part

# Function to combine CSV files from each folder into one file row-wise
def combine_csv_files(folder_path):
    output_folder = "C:\\Users\\aligh\\OneDrive\\Desktop\\dd8"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_data = []

    folder_list = sorted(os.listdir(folder_path), key=custom_key)  # Sort folders based on numerical values

    for folder_name in folder_list:
        folder_dir = os.path.join(folder_path, folder_name)

        if os.path.isdir(folder_dir):
            csv_files = [f for f in os.listdir(folder_dir) if f.endswith(".csv")]

            for file in csv_files:
                file_path = os.path.join(folder_dir, file)
                with open(file_path, 'r') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    data = list(csv_reader)
                    all_data.extend(data)

    output_file_path = os.path.join(output_folder, "all_part.csv")
    with open(output_file_path, 'w', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerows(all_data)

# Replace 'C:\\Users\\aligh\\OneDrive\\Desktop\\dd7' with your folder path containing folders with CSV files
combine_csv_files("C:\\Users\\aligh\\OneDrive\\Desktop\\dd7")


# In[ ]:




