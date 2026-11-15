#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "========================================================="
echo " STUDENT PERFORMANCE ANALYTICS & PREDICTION SYSTEM  "
echo "=================================================="
echo ""

echo "[1/5] Cleaning raw student datasets..."
py.exe -3.13 clean_data.py
echo "âœ” Datasets cleaned and saved!"
echo ""

echo "[2/5] Calculating Z-scores & predicting grades..."
py.exe -3.13 predict_grades.py
echo "âœ” Predictions saved to present_batch_predicted_grades.csv!"
echo ""

echo "[3/5] Generating performance visualizations..."
py.exe -3.13 visualize_data.py
echo "âœ” Visualizations complete!"
echo ""

echo "[4/5] Running grade prediction test cases..."
py.exe -3.13 run_tests.py
echo "âœ” Unit tests completed!"
echo ""

echo "[5/5] Profiling code execution..."
{
  echo "=== 1. PROFILING: clean_data.py ==="
  py.exe -3.13 -m cProfile -s tottime clean_data.py | head -n 30

  echo ""
  echo "=== 2. PROFILING: predict_grades.py ==="
  py.exe -3.13 -m cProfile -s tottime predict_grades.py | head -n 30

  echo ""
  echo "=== 3. PROFILING: visualize_data.py ==="
  py.exe -3.13 -m cProfile -s tottime visualize_data.py | head -n 30
} > report.txt
cat report.txt
echo "âœ” Profiling complete! Report saved to report.txt"
echo ""

echo "========================================================="
echo "PIPELINE FINISHED SUCCESSFULLY!              "
echo "===================================================="
