#!/bin/bash

echo "Running git lfs pull..."
git lfs pull

echo "Starting Streamlit app..."
streamlit run 1_Loan_Predictor.py
