
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import joblib
import math

# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "water_savings_model.joblib"
FEATURES_PATH = MODEL_DIR / "water_savings_features.joblib"

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"sqlite:///{(BASE_DIR / 'amanzi_impilo.db').as_posix()}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# --------------------------------------------------
# 2. DATABASE MODELS
# --------------------------------------------------

class SensorReading(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    source = db.Column(db.String(50), nullable=False, default="test")
    flow_litres = db.Column(db.Float)
    water_level_cm = db.Column(db.Float)
    ph = db.Column(db.Float)
    turbidity_ntu = db.Column(db.Float)
    tds_ppm = db.Column(db.Float)
    temperature_c = db.Column(db.Float)
    recorded_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )


class WaterPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    target = db.Column(db.String(100), nullable=False)
    predicted_litres_per_day = db.Column(
        db.Float, nullable=False
    )
    model_name = db.Column(
        db.String(100),
        nullable=False,
        default="water_savings_model"
    )
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )


# --------------------------------------------------
# 3. LOAD THE TRAINED MODEL
# --------------------------------------------------

if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    raise FileNotFoundError(
        "Model files not found. Train the model in Jupyter "
        "and save both files in the project's models folder."
    )

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

if not isinstance(feature_names, list) or not feature_names:
    raise ValueError(
        "The saved model feature list must be a non-empty list."
    )


# --------------------------------------------------
# 4. HOME AND HEALTH ENDPOINTS
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "application": "Amanzi Impilo ML API",
        "status": "running",
        "model": "Potable Water Savings",
        "unit": "litres per day",
        "endpoints": {
            "home": "GET /",
            "health": "GET /health",
            "features": "GET /features",
            "predict": "POST /predict",
            "save_reading": "POST /readings",
            "get_readings": "GET /readings",
            "get_predictions": "GET /predictions"
        }
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


# --------------------------------------------------
# 5. SAVE SENSOR READINGS
# --------------------------------------------------

@app.route("/readings", methods=["POST"])
def save_reading():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Send a valid JSON object."
        }), 400

    fields = [
        "flow_litres",
        "water_level_cm",
        "ph",
        "turbidity_ntu",
        "tds_ppm",
        "temperature_c"
    ]

    values = {}

    try:
        for field in fields:
            value = data.get(field)

            if value is None:
                values[field] = None
                continue

            if isinstance(value, bool):
                raise ValueError(
                    f"{field} must be a numeric value."
                )

            value = float(value)

            if not math.isfinite(value):
                raise ValueError(
                    f"{field} must be a finite number."
                )

            values[field] = value

        source = data.get("source", "test")

        if not isinstance(source, str) or not source.strip():
            return jsonify({
                "error": "source must be a non-empty string."
            }), 400

        reading = SensorReading(
            source=source.strip()[:50],
            **values
        )

        db.session.add(reading)
        db.session.commit()

        return jsonify({
            "message": "Sensor reading saved successfully.",
            "reading_id": reading.id
        }), 201

    except (TypeError, ValueError) as error:
        db.session.rollback()
        return jsonify({
            "error": "Invalid sensor input.",
            "details": str(error)
        }), 400

    except Exception:
        db.session.rollback()
        app.logger.exception("Could not save sensor reading")
        return jsonify({
            "error": "Database operation failed."
        }), 500


# --------------------------------------------------
# 6. RETRIEVE SENSOR READINGS
# --------------------------------------------------

@app.route("/readings", methods=["GET"])
def get_readings():
    try:
        readings = db.session.execute(
            db.select(SensorReading)
            .order_by(SensorReading.id.desc())
            .limit(100)
        ).scalars().all()

        return jsonify({
            "count": len(readings),
            "readings": [
                {
                    "id": r.id,
                    "source": r.source,
                    "flow_litres": r.flow_litres,
                    "water_level_cm": r.water_level_cm,
                    "ph": r.ph,
                    "turbidity_ntu": r.turbidity_ntu,
                    "tds_ppm": r.tds_ppm,
                    "temperature_c": r.temperature_c,
                    "recorded_at": (
                        r.recorded_at.isoformat()
                        if r.recorded_at else None
                    )
                }
                for r in readings
            ]
        })

    except Exception:
        app.logger.exception("Could not retrieve readings")
        return jsonify({
            "error": "Could not retrieve sensor readings."
        }), 500


# --------------------------------------------------
# 7. PREDICT POTABLE WATER SAVINGS
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Send a JSON object containing model inputs."
        }), 400

    inputs = data.get("features", data)

    if not isinstance(inputs, dict):
        return jsonify({
            "error": "features must be a JSON object."
        }), 400

    missing = [
        name for name in feature_names
        if name not in inputs
    ]

    if missing:
        return jsonify({
            "error": "Required model inputs are missing.",
            "missing_features": missing,
            "hint": "Call GET /features to see required inputs."
        }), 400

    try:
        row = pd.DataFrame(
            [[inputs[name] for name in feature_names]],
            columns=feature_names
        )

        for name in feature_names:
            value = row[name].iloc[0]

            if isinstance(value, (bool, np.bool_)):
                raise ValueError(
                    f"{name} must be numeric, not boolean."
                )

        row = row.apply(pd.to_numeric, errors="raise")

        if not np.isfinite(row.to_numpy(dtype=float)).all():
            return jsonify({
                "error": "All model inputs must be finite numbers."
            }), 400

        prediction = float(model.predict(row)[0])

        if not math.isfinite(prediction):
            raise ValueError(
                "The model returned a non-finite prediction."
            )

        estimated_savings = max(0.0, prediction)

        # Save the prediction to the database.
        record = WaterPrediction(
            target="potable_water_saved",
            predicted_litres_per_day=estimated_savings,
            model_name="water_savings_model"
        )

        db.session.add(record)
        db.session.commit()

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
            "prediction_id": record.id,
            "notice": (
                "Synthetic-data model prediction. The annual "
                "figure assumes a constant daily estimate and "
                "is not a validated annual forecast."
            )
        })

    except (TypeError, ValueError) as error:
        db.session.rollback()
        return jsonify({
            "error": "Invalid model input or prediction.",
            "details": str(error)
        }), 400

    except Exception:
        db.session.rollback()
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Check server logs."
        }), 500


# --------------------------------------------------
# 8. RETRIEVE PREDICTION HISTORY
# --------------------------------------------------

@app.route("/predictions", methods=["GET"])
def get_predictions():
    try:
        records = db.session.execute(
            db.select(WaterPrediction)
            .order_by(WaterPrediction.id.desc())
            .limit(100)
        ).scalars().all()

        return jsonify({
            "count": len(records),
            "predictions": [
                {
                    "id": r.id,
                    "target": r.target,
                    "predicted_litres_per_day":
                        r.predicted_litres_per_day,
                    "model_name": r.model_name,
                    "created_at": (
                        r.created_at.isoformat()
                        if r.created_at else None
                    )
                }
                for r in records
            ]
        })

    except Exception:
        app.logger.exception("Could not retrieve predictions")
        return jsonify({
            "error": "Could not retrieve prediction history."
        }), 500


# --------------------------------------------------
# 9. START THE APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    print("\nAmanzi Impilo API routes:")
    print(app.url_map)

    app.run(debug=True, port=5000)
