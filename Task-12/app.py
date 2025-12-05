from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "diabetes_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data.csv")

# Load model
model = joblib.load(MODEL_PATH)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Columns to exclude from dropdowns
exclude_cols = [
    "Age","Sex","Ethnicity","BMI","Waist_Circumference","Fasting_Blood_Glucose",
    "HbA1c","Blood_Pressure_Systolic","Blood_Pressure_Diastolic",
    "Cholesterol_Total","Cholesterol_HDL","Cholesterol_LDL",
    "GGT","Serum_Urate","Physical_Activity_Level","Dietary_Intake_Calories",
    "Alcohol_Consumption","Smoking_Status","Family_History_of_Diabetes",
    "Previous_Gestational_Diabetes"
]

# Features for dropdowns
dropdown_features = [col for col in df.columns if col not in exclude_cols + ["Diabetes_Risk"]]

# Create options for dropdowns
dropdown_options = {feature: df[feature].dropna().unique().tolist() for feature in dropdown_features}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Collect input data from form
        input_data = {}
        for feature in dropdown_features + exclude_cols:
            value = request.form.get(feature)
            input_data[feature] = value

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Convert numeric columns to float
        for col in exclude_cols:
            if col in input_df.columns:
                try:
                    input_df[col] = pd.to_numeric(input_df[col])
                except:
                    pass

        # Predict Diabetes Risk
        prediction = model.predict(input_df)[0]

        return render_template("index.html", dropdown_options=dropdown_options, prediction=prediction)

    return render_template("index.html", dropdown_options=dropdown_options, prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
