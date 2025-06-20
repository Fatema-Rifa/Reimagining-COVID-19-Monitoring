# COVID-19 Health Status Predictor

An AI-powered Streamlit application that analyzes patient vital signs (Oxygen level, Pulse rate, Body temperature) to predict potential COVID-19-related health abnormalities. The app integrates Google's Gemini AI to provide natural language health advice based on the input.

![App Screenshot](https://raw.githubusercontent.com/Fatema-Rifa/Reimagining-COVID-19-Monitoring/main/assests/app.png)


## Features

- Predicts health status (Normal or Abnormal) using a trained ML model
- Interactive sliders for vital sign input
- Natural language health insights powered by Gemini AI
- Displays prediction confidence
- Accepts optional symptom and medical history input
- Built-in medical disclaimer

---

## Project Structure

```

├── app.py                     # Streamlit application script
├── best\_model.pkl            # Trained classifier
├── scaler.pkl                 # Feature scaler
├── requirements.txt           # Dependency list
├── README.md                  # Project documentation
├── assets/
└── dataset/                   # Dataset for the app

````


## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Fatema-Rifa/Reimagining-COVID-19-Monitoring.git
cd Reimagining-COVID-19-Monitoring
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Gemini API Key

Option 1: As an environment variable

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

Option 2: Using `.streamlit/secrets.toml`

```toml
GEMINI_API_KEY = "your-gemini-api-key"
```

### 4. Run the Application

```bash
streamlit run app.py
```


## Machine Learning Overview

* Dataset columns: `Oxygen`, `PulseRate`, `Temperature`, `Result`
* Multiple models evaluated: Logistic Regression, KNN, SVM, Random Forest, XGBoost, CatBoost
* Best-performing model selected based on F1 Score
* Models and scaler exported via `joblib`


## Deployment

This app can be deployed on:

* Streamlit Cloud
* Render
* Heroku (with `Procfile`)

Ensure environment variables or `secrets.toml` are properly configured on the platform.


## Disclaimer

This application is intended for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a licensed healthcare provider for medical concerns.


## Contact

| Field      | Information |
|------------|-------------|
| **Author**   | Mst. Fatematuj Johora |
| **GitHub**   | [fatema-rifa](https://github.com/fatema-rifa) |
| **Portfolio** | [fatema-rifa.github.io](http://fatema-rifa.github.io/) |
| **Email**     | [mstfatematujjohora246@gmail.com](mailto:mstfatematujjohora246@gmail.com) |