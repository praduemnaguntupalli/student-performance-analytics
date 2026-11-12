
import pandas as pd
import numpy as np


class GradePredictor:

    def __init__(self):

        self.previous = pd.read_csv(
            "student_dataset_previous_batch.csv",
            header=None
        )

        self.current = pd.read_csv(
            "student_dataset_current_batch.csv",
            header=None
        )

        # Original column positions
        self.subjects = {
            "Calculus": {
                "previous_marks": [3, 4, 5, 6, 7],
                "previous_grade": 8,
                "current_marks": [3, 4, 5, 6, 7],
                "insert_position": 8
            },

            "DSA": {
                "previous_marks": [10, 11, 12, 13, 14],
                "previous_grade": 15,
                "current_marks": [9, 10, 11, 12, 13],
                "insert_position": 14
            },

            "Digital Systems": {
                "previous_marks": [17, 18, 19, 20, 21],
                "previous_grade": 22,
                "current_marks": [15, 16, 17, 18, 19],
                "insert_position": 20
            }
        }

    # --------------------------------
    # Calculate weighted marks
    # --------------------------------

    def calculate_marks(self, data, columns):

        marks = data.iloc[:, columns].apply(
            pd.to_numeric,
            errors="coerce"
        )

        score = (
            marks.iloc[:, 0] / 20 * 10
            + marks.iloc[:, 1] / 10 * 10
            + marks.iloc[:, 2] / 10 * 10
            + marks.iloc[:, 3] / 50 * 30
            + marks.iloc[:, 4] / 100 * 40
        )

        # Missing marks should not be treated as zero
        score[marks.isna().any(axis=1)] = np.nan

        return score

    # --------------------------------
    # Calculate Z-score
    # --------------------------------

    def calculate_z(self, scores):

        mean = scores.mean()
        sd = scores.std(ddof=0)

        if pd.isna(sd) or sd == 0:
            raise ValueError(
                "Standard deviation is zero. Cannot calculate Z-score."
            )

        z = (scores - mean) / sd

        return z, mean, sd

    # --------------------------------
    # Find average Z-score for grades
    # in previous batch
    # --------------------------------

    def find_grade_averages(self, subject, details):

        scores = self.calculate_marks(
            self.previous.iloc[2:],
            details["previous_marks"]
        ).reset_index(drop=True)

        grades = (
            self.previous.iloc[2:, details["previous_grade"]]
            .reset_index(drop=True)
            .astype("string")
            .str.strip()
            .str.upper()
        )

        data = pd.DataFrame({
            "Marks": scores,
            "Grade": grades
        })

        data = data.dropna(
            subset=["Marks", "Grade"]
        )

        data = data[data["Grade"] != ""]

        z, mean, sd = self.calculate_z(data["Marks"])

        data["Z"] = z

        grade_average = data.groupby("Grade")["Z"].mean()

        print("\n--------------------------------")
        print(subject, "PREVIOUS BATCH")
        print("--------------------------------")

        print("Mean:", round(mean, 2))
        print("Standard Deviation:", round(sd, 2))

        print("\nAverage Z-score of each grade:")
        print(grade_average.round(3))

        return grade_average

    # --------------------------------
    # Predict grade using closest Zavg
    # --------------------------------

    def predict(self, z, grade_average):

        if pd.isna(z):
            return "N/A"

        difference = (grade_average - z).abs()

        closest_grade = difference.idxmin()

        return closest_grade

    # --------------------------------
    # Main prediction
    # --------------------------------

    def run(self):

        # Store predictions separately
        predictions = {}

        for subject, details in self.subjects.items():

            print("\n================================")
            print("PROCESSING:", subject)
            print("================================")

            # Previous batch grade-wise Z averages
            grade_average = self.find_grade_averages(
                subject,
                details
            )

            # Current batch weighted marks
            current_scores = self.calculate_marks(
                self.current.iloc[2:],
                details["current_marks"]
            ).reset_index(drop=True)

            # Current batch Z-scores
            current_z, current_mean, current_sd = (
                self.calculate_z(current_scores)
            )

            # Predict each student's grade
            predicted_grades = current_z.apply(
                lambda z: self.predict(z, grade_average)
            )

            predictions[subject] = predicted_grades.tolist()

            print("\nCurrent Mean:", round(current_mean, 2))
            print("Current SD:", round(current_sd, 2))

            print("\nPredicted grade distribution:")
            print(predicted_grades.value_counts())

        # --------------------------------
        # Preserve original CSV structure
        # --------------------------------

        result = self.current.copy()

        # Insert from right to left so positions
        # of original columns do not change

        for subject in reversed(list(self.subjects.keys())):

            details = self.subjects[subject]

            position = details["insert_position"]

            # Two header rows followed by student predictions
            new_column = [
                subject,
                "Predicted Grade"
            ] + predictions[subject]

            result.insert(
                position,
                subject + "_Predicted_Grade",
                new_column
            )

        # Save complete CSV with original two headers
        result.to_csv(
            "present_batch_predicted_grades.csv",
            index=False,
            header=False
        )

        print("\n================================")
        print("PREDICTION COMPLETED SUCCESSFULLY")
        print("================================")

        print("Output file:")
        print("present_batch_predicted_grades.csv")


# --------------------------------
# Execute
# --------------------------------

if __name__ == "__main__":

    predictor = GradePredictor()

    predictor.run()
