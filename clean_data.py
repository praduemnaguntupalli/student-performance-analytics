
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

CURRENT_FILE = BASE_DIR / "student_dataset_current_batch.csv"
PREVIOUS_FILE = BASE_DIR / "student_dataset_previous_batch.csv"

CURRENT_OUTPUT = BASE_DIR / "cleaned_current.csv"
PREVIOUS_OUTPUT = BASE_DIR / "cleaned_previous.csv"


subjects = ["calculus", "dsa", "digital_systems"]

assessments = [
    "attendance_(%)",
    "assignments_(20)",
    "quiz_1_(10)",
    "quiz_2_(10)",
    "midsem_(50)",
    "endsem_(100)"
]

current_columns = ["Student_Name", "Roll_No"]

for subject in subjects:
    for assessment in assessments:
        current_columns.append(subject + "_" + assessment)


previous_columns = ["Student_Name", "Roll_No"]

for subject in subjects:
    for assessment in assessments:
        previous_columns.append(subject + "_" + assessment)

    previous_columns.append(subject + "_grade")


def clean_data(filename, columns):

    data = pd.read_csv(
        filename,
        header=None,
        dtype=str
    )

    data = data.iloc[2:].copy()

    if data.shape[1] != len(columns):
        raise ValueError(
            f"\nColumn mismatch in {filename.name}!"
            f"\nExpected columns: {len(columns)}"
            f"\nFound columns: {data.shape[1]}"
        )

    data.columns = columns

    data["Student_Name"] = data["Student_Name"].str.strip()
    data["Roll_No"] = data["Roll_No"].str.strip()

    data = data.dropna(
        subset=["Student_Name", "Roll_No"]
    )

    data = data[
        (data["Student_Name"] != "") &
        (data["Roll_No"] != "")
    ]

    data = data.drop_duplicates(
        subset=["Roll_No"]
    )

    for column in data.columns[2:]:

        if column.endswith("_grade"):

            data[column] = (
                data[column]
                .fillna("")
                .str.strip()
                .str.upper()
            )

        else:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    data = data.reset_index(drop=True)

    return data


current = clean_data(
    CURRENT_FILE,
    current_columns
)

previous = clean_data(
    PREVIOUS_FILE,
    previous_columns
)


current.to_csv(
    CURRENT_OUTPUT,
    index=False
)

previous.to_csv(
    PREVIOUS_OUTPUT,
    index=False
)


print("\n" + "=" * 50)
print("CURRENT BATCH")
print("=" * 50)

print(current.head())

print("\nCurrent batch shape:", current.shape)
current.info()


print("\n" + "=" * 50)
print("PREVIOUS BATCH")
print("=" * 50)

print(previous.head())

print("\nPrevious batch shape:", previous.shape)
previous.info()


print("\n" + "=" * 50)
print("DATA CLEANING COMPLETED SUCCESSFULLY!")
print("=" * 50)

print("\nGenerated files:")
print(CURRENT_OUTPUT)
print(PREVIOUS_OUTPUT)
