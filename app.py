
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)

MODEL_PATH = Path("models/water_savings_model.joblib")
FEATURES_PATH = Path("models/water_savings_features.joblib")

# Load the model once when Flask starts
if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    raise FileNotFoundError(
        "Model files not found. Train the model in Jupyter first."
    )

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "application": "Amanzi Impilo ML API",
        "status": "running",
        "model": "Potable Water Savings",
        "unit": "litres per day",
        "prediction_endpoint": "/predict",
        "method": "POST"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "number_of_features": len(feature_names)
    })


@app.route("/features", methods=["GET"])
def features():
    return jsonify({
        "required_features": feature_names,
        "count": len(feature_names)
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Send a JSON object containing input features."
            }), 400

        # Accept {"features": {...}} or a direct feature dictionary
        inputs = data.get("features", data)

        missing = [
            name for name in feature_names
            if name not in inputs
        ]

        if missing:
            return jsonify({
                "error": "Some required model inputs are missing.",
                "missing_features": missing,
                "hint": "Call GET /features to see all required inputs."
            }), 400

        # Keep exactly the feature names and order used during training
        row = pd.DataFrame(
            [[inputs[name] for name in feature_names]],
            columns=feature_names
        )

        row = row.apply(pd.to_numeric, errors="raise")

        if row.isnull().any().any():
            return jsonify({
                "error": "Input features cannot contain missing values."
            }), 400

        prediction = float(model.predict(row)[0])

        # Water savings cannot physically be negative.
        # Preserve the raw model estimate for transparency.
        estimated_savings = max(0.0, prediction)

        return jsonify({
            "project": "Amanzi Impilo",
            "prediction": {
                "target": "potable_water_saved",
                "estimated_litres_per_day": round(
                    estimated_savings, 2
                ),
                "raw_model_estimate_litres_per_day": round(
                    prediction, 2
                ),
                "illustrative_365_day_saving_litres": round(
                    estimated_savings * 365, 2
                )
            },
            "notice": (
                "Prediction from a model trained on synthetic data. "
                "The 365-day figure assumes the daily estimate remains "
                "constant; it is not a validated annual forecast."
            )
        })

    except (TypeError, ValueError) as error:
        return jsonify({
            "error": "Invalid input values.",
            "details": str(error)
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Check the server logs."
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
