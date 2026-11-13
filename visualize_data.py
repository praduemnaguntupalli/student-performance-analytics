
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. READ THE DATASETS
# --------------------------------------------------

folder = Path(__file__).resolve().parent

current = pd.read_csv(folder / "cleaned_current.csv")
previous = pd.read_csv(folder / "cleaned_previous.csv")

# Standardise column names
current.columns = current.columns.str.strip().str.lower().str.replace(" ", "_")
previous.columns = previous.columns.str.strip().str.lower().str.replace(" ", "_")

subjects = ["calculus", "dsa", "digital_systems"]

subject_names = {
    "calculus": "Calculus",
    "dsa": "DSA",
    "digital_systems": "Digital Systems"
}

colours = {
    "calculus": "blue",
    "dsa": "red",
    "digital_systems": "green"
}

grades = ["AA", "AB", "BB", "BC", "CC", "DX", "FR"]


# --------------------------------------------------
# 2. CALCULATE WEIGHTED AVERAGE
# --------------------------------------------------

def calculate_marks(data, subject):

    assignments = pd.to_numeric(
        data[subject + "_assignments_(20)"], errors="coerce"
    )

    quiz1 = pd.to_numeric(
        data[subject + "_quiz_1_(10)"], errors="coerce"
    )

    quiz2 = pd.to_numeric(
        data[subject + "_quiz_2_(10)"], errors="coerce"
    )

    midsem = pd.to_numeric(
        data[subject + "_midsem_(50)"], errors="coerce"
    )

    endsem = pd.to_numeric(
        data[subject + "_endsem_(100)"], errors="coerce"
    )

    # Weighted marks out of 100
    total = (
        (assignments / 20) * 10
        + (quiz1 / 10) * 10
        + (quiz2 / 10) * 10
        + (midsem / 50) * 30
        + (endsem / 100) * 40
    )

    # If any assessment is missing, keep the total missing
    return total.where(
        assignments.notna()
        & quiz1.notna()
        & quiz2.notna()
        & midsem.notna()
        & endsem.notna()
    )


# Calculate marks for all three subjects
for subject in subjects:

    current[subject + "_average"] = calculate_marks(current, subject)

    previous[subject + "_average"] = calculate_marks(previous, subject)


# --------------------------------------------------
# 3. FIND GRADES
# --------------------------------------------------

def get_grades(data, subject):

    grade_column = subject + "_grade"

    if grade_column in data.columns:

        result = data[grade_column].astype(str).str.strip().str.upper()

    else:

        marks = data[subject + "_average"]

        result = pd.cut(
            marks,
            bins=[-np.inf, 43.5, 49.5, 57.5, 65.5, 74.5, 84.5, np.inf],
            labels=["FR", "DX", "CC", "BC", "BB", "AB", "AA"]
        ).astype("object")

    # Attendance below 75% is assigned DX
    attendance_column = subject + "_attendance_(%)"

    if attendance_column in data.columns:

        attendance = pd.to_numeric(
            data[attendance_column], errors="coerce"
        )

        result.loc[attendance < 75] = "DX"

    return result


# --------------------------------------------------
# GRAPH 1: SUBJECT AVERAGE MARKS
# --------------------------------------------------

def subject_average_graph():

    averages = []

    for subject in subjects:
        averages.append(current[subject + "_average"].mean())

    plt.figure(figsize=(8, 5))

    plt.bar(
        ["Calculus", "DSA", "Digital Systems"],
        averages,
        color=["blue", "red", "green"]
    )

    plt.title("Average Marks of Three Subjects")
    plt.xlabel("Subjects")
    plt.ylabel("Average Marks")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# GRAPH 2: ATTENDANCE VS AVERAGE MARKS
# --------------------------------------------------

def attendance_graph():

    plt.figure(figsize=(8, 5))

    for subject in subjects:

        attendance_column = subject + "_attendance_(%)"

        if attendance_column not in current.columns:
            continue

        attendance = pd.to_numeric(
            current[attendance_column], errors="coerce"
        )

        marks = current[subject + "_average"]

        plt.scatter(
            attendance,
            marks,
            color=colours[subject],
            label=subject_names[subject],
            alpha=0.6
        )

    plt.axvline(75, color="black", linestyle="--", label="75% Attendance")

    plt.title("Attendance vs Average Marks")
    plt.xlabel("Attendance (%)")
    plt.ylabel("Average Marks")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# GRAPH 3: ASSESSMENT-WISE COMPARISON
# --------------------------------------------------

def assessment_graph():

    assessments = [
        "assignments_(20)",
        "quiz_1_(10)",
        "quiz_2_(10)",
        "midsem_(50)",
        "endsem_(100)"
    ]

    weights = [10, 10, 10, 30, 40]

    values = []

    for subject in subjects:

        subject_values = []

        for assessment, weight in zip(assessments, weights):

            column = subject + "_" + assessment

            marks = pd.to_numeric(current[column], errors="coerce")

            maximum = int(assessment.split("(")[1].replace(")", ""))

            contribution = (marks / maximum) * weight

            subject_values.append(contribution.mean())

        values.append(subject_values)

    x = np.arange(len(assessments))
    width = 0.25

    plt.figure(figsize=(10, 5))

    for i, subject in enumerate(subjects):

        plt.bar(
            x + i * width,
            values[i],
            width,
            label=subject_names[subject],
            color=colours[subject]
        )

    plt.xticks(
        x + width,
        ["Assignments", "Quiz 1", "Quiz 2", "Midsem", "Endsem"]
    )

    plt.title("Assessment-wise Subject Comparison")
    plt.xlabel("Assessments")
    plt.ylabel("Average Weighted Contribution")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# GRAPH 4: GRADE DISTRIBUTION
# --------------------------------------------------

def grade_distribution():

    grade_counts = {}

    for subject in subjects:

        student_grades = get_grades(current, subject)

        grade_counts[subject_names[subject]] = [
            (student_grades == grade).sum()
            for grade in grades
        ]

    x = np.arange(len(grades))
    width = 0.25

    plt.figure(figsize=(9, 5))

    for i, subject in enumerate(subjects):

        plt.bar(
            x + i * width,
            grade_counts[subject_names[subject]],
            width,
            label=subject_names[subject],
            color=colours[subject]
        )

    plt.xticks(x + width, grades)

    plt.title("Grade Distribution of Three Subjects")
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# GRAPH 5: MARKS HISTOGRAM COMPARISON
# --------------------------------------------------

def marks_histogram():

    bins = [0, 20, 40, 60, 80, 100]

    counts = []

    for subject in subjects:

        marks = current[subject + "_average"].dropna()

        histogram, _ = np.histogram(marks, bins=bins)

        counts.append(histogram)

    x = np.arange(len(bins) - 1)
    width = 0.25

    plt.figure(figsize=(9, 5))

    for i, subject in enumerate(subjects):

        plt.bar(
            x + i * width,
            counts[i],
            width,
            label=subject_names[subject],
            color=colours[subject]
        )

    plt.xticks(
        x + width,
        ["0-20", "20-40", "40-60", "60-80", "80-100"]
    )

    plt.title("Marks Distribution of Three Subjects")
    plt.xlabel("Marks Range")
    plt.ylabel("Number of Students")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# GRAPH 6: Z-SCORE VS GRADE SCATTER PLOT
# --------------------------------------------------

def z_average_vs_grade():

    plt.figure(figsize=(11, 6))

    grade_positions = {
        grade: i for i, grade in enumerate(grades)
    }

    # Use a fixed random generator for consistent jitter
    random_generator = np.random.default_rng(42)

    for subject in subjects:

        marks = previous[subject + "_average"]

        # Calculate mean and standard deviation from previous batch
        mean = marks.mean()
        std = marks.std()

        if pd.isna(std) or std == 0:
            continue

        # Calculate individual student Z-scores
        z_scores = (marks - mean) / std

        # Get the grade of each previous-batch student
        student_grades = get_grades(previous, subject)

        x_values = []
        y_values = []

        for grade in grades:

            # Select students belonging to this grade
            selected = student_grades == grade

            selected_z = z_scores[selected].dropna()

            for z in selected_z:

                # Small horizontal shift to avoid overlapping circles
                jitter = random_generator.uniform(-0.12, 0.12)

                x_values.append(grade_positions[grade] + jitter)
                y_values.append(z)

        # Plot individual students using circles
        plt.scatter(
            x_values,
            y_values,
            color=colours[subject],
            label=subject_names[subject],
            marker="o",
            s=45,
            alpha=0.65
        )

    plt.axhline(
        0,
        color="black",
        linestyle="--",
        label="Z-score = 0"
    )

    plt.xticks(
        range(len(grades)),
        grades
    )

    plt.title("Individual Student Z-score vs Grade")
    plt.xlabel("Grades")
    plt.ylabel("Individual Student Z-score")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 7. DISPLAY ALL GRAPHS
# --------------------------------------------------

def main():

    subject_average_graph()

    attendance_graph()

    assessment_graph()

    grade_distribution()

    marks_histogram()

    z_average_vs_grade()


if __name__ == "__main__":

    main()
