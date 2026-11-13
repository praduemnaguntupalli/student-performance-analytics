+#!/bin/bash
cd "$(dirname "$0")"
echo "========================================================="
echo "   🎓 STUDENT PERFORMANCE ANALYTICS & PREDICTION SYSTEM  "
echo "========================================================="
echo ""

echo "▶ [1/3] Cleaning raw student datasets..."
py.exe -3.13 clean_data.py
echo "✔ Datasets cleaned and saved!"
echo ""

echo "▶ [2/3] Calculating Z-scores & predicting grades..."
py.exe -3.13 predict_grades.py
echo "✔ Predictions saved to present_batch_predicted_grades.csv!"
echo ""

echo "▶ [3/3] Generating performance visualizations..."
py.exe -3.13 visualize_data.py
echo "✔ Visualizations complete!"
echo ""

echo "========================================================="
echo "         🚀 PIPELINE FINISHED SUCCESSFULLY!              "
echo "========================================================="