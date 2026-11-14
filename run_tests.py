import pandas as pd
from predict_grades import GradePredictor

predictor = GradePredictor()

# sample average z-scores for grades
avg_z = pd.Series({
    "AA": 1.5,
    "AB": 0.8,
    "BB": 0.0,
    "BC": -0.5,
    "CC": -1.0,
    "FR": -2.0
})

# test cases: (z_score, expected_grade)
tests = [
    (1.6, "AA"),
    (0.75, "AB"),
    (0.1, "BB"),
    (-0.4, "BC"),
    (-1.1, "CC"),
    (-2.2, "FR"),
    (float("nan"), "N/A")
]

passed = 0

for z, expected in tests:
    got = predictor.predict(z, avg_z)
    if got == expected:
        print("Test passed for z =", z, "-> Grade:", got)
        passed += 1
    else:
        print("Test failed for z =", z, "-> Expected:", expected, "Got:", got)

print("\nTotal passed:", passed, "out of", len(tests))
