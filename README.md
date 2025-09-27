# SmartTB-Diagnosis-DecisionSupport

An intelligent clinical decision-support system designed to help healthcare providers diagnose tuberculosis (TB) accurately and reduce the risk of multidrug-resistant TB (MDR-TB).

## Overview
SmartTB leverages a Bayesian network model to assess patient symptoms and prior risk factors, producing disease probability estimates in real-time. The system offers an intuitive interface for symptom selection and visualizes probabilities to guide clinical decision-making.

## Key Features
- Symptom-based disease probability calculation for TB, Pneumonia, Bronchitis, and Lung Cancer.
- Interactive sliders to adjust disease priors based on local prevalence or patient history.
- Dynamic probability bars with clear visual feedback.
- Designed to minimize diagnostic errors and support early intervention for MDR-TB.

## Technologies
- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Flask (Python)
- **Modeling:** Bayesian Network for probabilistic reasoning
- **API:** JSON-based communication between frontend and backend

## Usage
1. Open the web interface.
2. Select relevant patient symptoms.
3. Adjust disease priors if needed.
4. View the dynamically updated probability estimates for each condition.
5. Use the results to support clinical decision-making.

## Goal
To provide healthcare providers with a rapid, reliable, and interpretable decision-support tool that enhances TB diagnosis and helps prevent the development and spread of MDR-TB.
