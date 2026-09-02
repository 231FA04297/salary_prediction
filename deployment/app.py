
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# EXPECTED FEATURES
# ============================================================

EXPECTED_FEATURES = list(
    model.feature_names_in_
)


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Salary Prediction API",
        "status": "running",
        "model": "Best Salary Prediction Model",
        "number_of_features": len(
            EXPECTED_FEATURES
        )
    })


# ============================================================
# PREDICT
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        # ----------------------------------------------------
        # ORIGINAL USER INPUT
        # ----------------------------------------------------

        age = float(
            data["Age"]
        )

        years_experience = float(
            data["Years of Experience"]
        )

        gender = data["Gender"]

        education = data["Education Level"]

        job_title = data["Job Title"]


        # ----------------------------------------------------
        # FEATURE ENGINEERING
        # ----------------------------------------------------

        age_experience_ratio = (
            age /
            (years_experience + 1)
        )

        experience_squared = (
            years_experience ** 2
        )

        age_squared = (
            age ** 2
        )

        age_experience = (
            age *
            years_experience
        )


        # ----------------------------------------------------
        # CREATE ORIGINAL DATAFRAME
        # ----------------------------------------------------

        input_df = pd.DataFrame({

            "Age": [age],

            "Years of Experience":
                [years_experience],

            "Age_Experience_Ratio":
                [age_experience_ratio],

            "Experience_Squared":
                [experience_squared],

            "Age_Squared":
                [age_squared],

            "Age_Experience":
                [age_experience],

            "Gender":
                [gender],

            "Education Level":
                [education],

            "Job Title":
                [job_title]
        })


        # ----------------------------------------------------
        # ONE-HOT ENCODING
        # ----------------------------------------------------

        input_df = pd.get_dummies(
            input_df,
            columns=[
                "Gender",
                "Education Level",
                "Job Title"
            ],
            prefix=[
                "categorical__Gender",
                "categorical__Education Level",
                "categorical__Job Title"
            ]
        )


        # ----------------------------------------------------
        # RENAME NUMERIC FEATURES
        # ----------------------------------------------------

        numeric_mapping = {

            "Age":
                "numeric__Age",

            "Years of Experience":
                "numeric__Years of Experience",

            "Age_Experience_Ratio":
                "numeric__Age_Experience_Ratio",

            "Experience_Squared":
                "numeric__Experience_Squared",

            "Age_Squared":
                "numeric__Age_Squared",

            "Age_Experience":
                "numeric__Age_Experience"
        }


        input_df = input_df.rename(
            columns=numeric_mapping
        )


        # ----------------------------------------------------
        # ADD MISSING FEATURES
        # ----------------------------------------------------

        for feature in EXPECTED_FEATURES:

            if feature not in input_df.columns:

                input_df[feature] = 0


        # ----------------------------------------------------
        # REMOVE EXTRA FEATURES
        # ----------------------------------------------------

        input_df = input_df[
            EXPECTED_FEATURES
        ]


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            input_df
        )


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify({

            "Age":
                age,

            "Years of Experience":
                years_experience,

            "Gender":
                gender,

            "Education Level":
                education,

            "Job Title":
                job_title,

            "predicted_salary":
                round(
                    float(prediction[0]),
                    2
                )
        })


    except Exception as e:

        return jsonify({

            "error":
                str(e)

        }), 400


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
