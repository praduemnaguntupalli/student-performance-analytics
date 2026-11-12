
import pandas as pd
import numpy as np

# Reading the datasets
current = pd.read_csv("student_dataset_current_batch.csv", header=None)

previous = pd.read_csv("student_dataset_previous_batch.csv", header=None)


# Function to clean the dataset
def clean_data(data):

    # Fill subject names across the columns
    data.iloc[0] = data.iloc[0].ffill()

    # Creating column names
    columns = ["Student_Name", "Roll_No"]

    for subject, exam in zip(data.iloc[0, 2:], data.iloc[1, 2:]):
        columns.append(
            str(subject).strip().lower() + "_" +
            str(exam).strip().lower().replace(" ", "_")
        )

    # Removing the first two header rows
    data = data.iloc[2:].copy()

    # Assigning column names
    data.columns = columns

    # Removing rows without student details
    data = data.dropna(subset=["Student_Name", "Roll_No"])

    # Removing duplicate roll numbers
    data = data.drop_duplicates(subset=["Roll_No"])

    # Removing extra spaces
    data["Student_Name"] = data["Student_Name"].str.strip()
    data["Roll_No"] = data["Roll_No"].str.strip()

    # Converting marks into numbers
    for column in data.columns[2:]:
        if "grade" not in column:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        else:
            data[column] = data[column].str.strip()
    return data


# Cleaning both datasets
current = clean_data(current)
previous = clean_data(previous)


# Saving cleaned datasets
current.to_csv("cleaned_current.csv", index=False)
previous.to_csv("cleaned_previous.csv", index=False)


# Displaying the cleaned data
print("CURRENT BATCH")
print(current.head())

print("\nPREVIOUS BATCH")
print(previous.head())

print("\nCurrent batch information:")
current.info()

print("\nPrevious batch information:")
previous.info()

print("\nCleaning completed!")
