# Anemia-checker-using-ML

This project is a web application designed to predict anemia in patients based on their blood pixel values and hemoglobin levels. The application uses logistic regression for prediction and includes data visualization features for analysis.

## Overview
Anemia Checker is a tool that leverages machine learning to assist in diagnosing anemia. Users can upload a CSV file with patient data, visualize the data distributions, and make predictions about whether a patient has anemia based on input features.

## Features
- **Data Upload:** Upload CSV files containing patient data.
- **Data Viewing:** View the uploaded data in a tabular format.
- **Data Visualization:** Visualize distributions of various features with histograms and count plots.
- **Anemia Prediction:** Predict anemia based on input values for Red Pixel, Green Pixel, Blue Pixel, and Hb (Hemoglobin).

## Usage
1. **Upload Your CSV File:**
    - Use the file uploader to upload a CSV file containing your data. The CSV file should have columns for Red Pixel, Green Pixel, Blue Pixel, Hb (Hemoglobin), and Anaemic.

2. **View Data:**
    - Navigate to the 'View Data' section to see the uploaded data in a tabular format.

3. **Visualization:**
    - Navigate to the 'Visualization' section to view various plots for data analysis:
        - Histograms for Red Pixel, Green Pixel, and Blue Pixel.
        - KDE plot for Hb.
        - Count plot for the Anaemic column.

4. **Prediction:**
    - Navigate to the 'Prediction' section.
    - Input the values for Red Pixel, Green Pixel, Blue Pixel, and Hb.
    - Click the 'Predict' button to get the prediction result.

## Dependencies
- pandas
- seaborn
- matplotlib
- streamlit
- scikit-learn
