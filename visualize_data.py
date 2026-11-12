from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class StudentVisualizer:

    def __init__(self, csv_filename="cleaned_current.csv"):

        # Search the script folder and any subfolders for the CSV file
        base_dir = Path(__file__).resolve().parent
        matches = list(base_dir.rglob(csv_filename))

        if not matches:
            available_csvs = [
                str(p.relative_to(base_dir)) for p in base_dir.rglob("*.csv")
            ]
            raise FileNotFoundError(
                f"Could not find '{csv_filename}' inside '{base_dir}'.\n"
                f"CSV files found in project: {available_csvs}"
            )

        # Read cleaned current batch
        self.data = pd.read_csv(matches[0])

        # Standardise column names
        self.data.columns = (
            self.data.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )

        # Subject names and display labels
        self.subjects = [
            "calculus",
            "dsa",
            "digital_systems"
        ]
        self.subject_labels = [
            "Calculus",
            "DSA",
            "Digital Systems"
        ]

        # Calculate weighted marks
        self.calculate_all_marks()

    # --------------------------------
    # Find a column using its name
    # --------------------------------

    def get_column(self, subject, assessment):

        possible_names = [
            subject + "_" + assessment,
            subject + "_" + assessment + "_(20)",
            subject + "_" + assessment + "_(10)",
            subject + "_" + assessment + "_(50)",
            subject + "_" + assessment + "_(100)"
        ]

        for name in possible_names:
            if name in self.data.columns:
                return name

        raise ValueError(
            "Column not found: " + subject + "_" + assessment
            + "\nAvailable columns:\n"
            + str(self.data.columns.tolist())
        )

    # --------------------------------
    # Calculate weighted marks
    # --------------------------------

    def calculate_marks(self, subject):

        assignment = self.get_column(subject, "assignments")
        quiz1 = self.get_column(subject, "quiz_1")
        quiz2 = self.get_column(subject, "quiz_2")
        midsem = self.get_column(subject, "midsem")
        endsem = self.get_column(subject, "endsem")

        marks = self.data[
            [assignment, quiz1, quiz2, midsem, endsem]
        ].apply(pd.to_numeric, errors="coerce")

        total = (
            marks.iloc[:, 0] / 20 * 10
            + marks.iloc[:, 1] / 10 * 10
            + marks.iloc[:, 2] / 10 * 10
            + marks.iloc[:, 3] / 50 * 30
            + marks.iloc[:, 4] / 100 * 40
        )

        total[marks.isna().any(axis=1)] = np.nan

        return total

    # --------------------------------
    # Calculate marks for all subjects
    # --------------------------------

    def calculate_all_marks(self):

        for subject in self.subjects:
            self.data[subject + "_average"] = self.calculate_marks(subject)

    # --------------------------------
    # Graph 1: Subject average comparison
    # --------------------------------

    def subject_average_graph(self):

        averages = [
            self.data[subject + "_average"].mean()
            for subject in self.subjects
        ]

        plt.figure(figsize=(12, 6))

        bars = plt.bar(self.subject_labels, averages)
        plt.bar_label(bars, fmt="%.2f", padding=3)

        plt.title("Average Weighted Marks Comparison")
        plt.xlabel("Subjects")
        plt.ylabel("Average Marks out of 100")

        plt.ylim(0, 100)
        plt.tight_layout()
        plt.show()

    # --------------------------------
    # Graph 2: Attendance vs marks
    # --------------------------------

    def attendance_graph(self):

        attendance_columns = [
            col for col in self.data.columns
            if "attendance" in col
        ]

        if not attendance_columns:
            print("Attendance column not found!")
            return

        attendance = pd.to_numeric(
            self.data[attendance_columns[0]],
            errors="coerce"
        )

        plt.figure(figsize=(12, 6))

        for subject in self.subjects:
            label = subject.replace("_", " ").title()
            plt.scatter(
                attendance,
                self.data[subject + "_average"],
                label=label,
                alpha=0.75
            )

        plt.title("Attendance vs Student Performance")
        plt.xlabel("Attendance (%)")
        plt.ylabel("Weighted Marks out of 100")

        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.6)
        plt.tight_layout()
        plt.show()

    # --------------------------------
    # Graph 3: Assessment Performance Comparison
    # --------------------------------

    def assessment_graph(self):

        assessments = [
            ("assignments", 20, 10, "Assignment"),
            ("quiz_1", 10, 10, "Quiz 1"),
            ("quiz_2", 10, 10, "Quiz 2"),
            ("midsem", 50, 30, "Midsem"),
            ("endsem", 100, 40, "Endsem")
        ]

        x = np.arange(len(assessments))
        width = 0.25

        plt.figure(figsize=(12, 6))

        for i, (subject, label) in enumerate(zip(self.subjects, self.subject_labels)):
            weighted_avgs = []

            for col_key, max_marks, weight, _ in assessments:
                column = self.get_column(subject, col_key)
                marks = pd.to_numeric(self.data[column], errors="coerce")
                weighted_avgs.append((marks / max_marks * weight).mean())

            plt.bar(x + (i - 1) * width, weighted_avgs, width, label=label)

        plt.title("Assessment Performance Comparison")
        plt.xlabel("Assessment Components")
        plt.ylabel("Average Weighted Contribution")
        plt.xticks(x, [item[3] for item in assessments])

        plt.legend(loc="upper left")
        plt.grid(axis="y", alpha=0.6)
        plt.tight_layout()
        plt.show()

    # --------------------------------
    # Graph 4: Grade Distribution Across Subjects
    # --------------------------------

    def grade_distribution(self):

        grade_order = ["AA", "AB", "BB", "BC", "CC", "DX", "FR"]
        x = np.arange(len(grade_order))
        width = 0.25

        plt.figure(figsize=(12, 6))

        for i, (subject, label) in enumerate(zip(self.subjects, self.subject_labels)):
            grade_col = subject + "_grade"

            if grade_col in self.data.columns:
                grades = self.data[grade_col].astype(str).str.strip().str.upper()
            else:
                marks = self.data[subject + "_average"]
                grades = pd.cut(
                    marks,
                    bins=[-np.inf, 43.5, 49.5, 57.5, 65.5, 74.5, 84.5, np.inf],
                    labels=["FR", "DX", "CC", "BC", "BB", "AB", "AA"]
                )

            counts = grades.value_counts().reindex(grade_order, fill_value=0)
            plt.bar(x + (i - 1) * width, counts.values, width, label=label)

        plt.title("Grade Distribution Across Subjects")
        plt.xlabel("Grades")
        plt.ylabel("Number of Students")
        plt.xticks(x, grade_order)

        plt.legend(loc="upper right")
        plt.grid(axis="y", alpha=0.6)
        plt.tight_layout()
        plt.show()

    # --------------------------------
    # Run all graphs
    # --------------------------------

    def run(self):

        print("\nGenerating visualizations...")

        self.subject_average_graph()
        self.attendance_graph()
        self.assessment_graph()
        self.grade_distribution()

        print("\nAll graphs generated successfully!")


# --------------------------------
# Main program
# --------------------------------

if __name__ == "__main__":

    visualizer = StudentVisualizer()
    visualizer.run()
