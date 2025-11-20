"""
Flask API for Ingredient Expiry Prediction
This API accepts ingredient data including storage type and returns predicted hours until expiry
and a freshness classification using a Random Forest model.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import pandas as pd

# ---------------------------
# Flask setup
# ---------------------------
app = Flask(__name__)
CORS(app)  # Enable CORS for React Native or frontend apps

# ---------------------------
# Model loading
# ---------------------------
MODEL_PATH = os.getenv('MODEL_PATH', 'expiry_predictor_model.joblib')
model = None

def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
        return True
    except FileNotFoundError:
        print(f"⚠️ Model file not found at {MODEL_PATH}")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

model_loaded = load_model()

# ---------------------------
# Prediction endpoint
# ---------------------------
@app.route('/predict', methods=['POST'])
def predict():
    """
    Request JSON:
    {
        "temperature": float (°C),
        "humidity": float (%),
        "time_in_refrigerator": float (hours),
        "ingredient_type": str ("BEEF", "CHEESE", etc.),
        "storage_type": str ("FRIDGE", "FREEZER", "PANTRY")
    }

    Response JSON:
    {
        "hours_until_expiry": float,
        "classification": "Fresh" | "Stale" | "Expired"
    }
    """
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    required_fields = ['temperature', 'humidity', 'time_in_refrigerator', 'ingredient_type', 'storage_type']

    # Validate required fields
    if not data:
        return jsonify({"error": "No data provided"}), 400
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        # Prepare features as a DataFrame (pipeline handles encoding)
        features = pd.DataFrame([{
            'temperature': float(data['temperature']),
            'humidity': float(data['humidity']),
            'time_in_inventory': float(data['time_in_refrigerator']),
            'ingredient_type': str(data['ingredient_type']),
            'storage_type': str(data['storage_type']).upper()
        }])

        # Predict hours until expiry
        predicted_hours = float(model.predict(features)[0])

        # Map to Fresh/Stale/Expired
        if predicted_hours <= 0:
            classification = "Expired"
        elif predicted_hours <= 24:
            classification = "Stale"
        else:
            classification = "Fresh"

        return jsonify({
            "hours_until_expiry": predicted_hours,
            "classification": classification
        })

    except ValueError as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction error: {e}"}), 500

# ---------------------------
# Health check
# ---------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded
    })

# ---------------------------
# Home endpoint
# ---------------------------
@app.route('/')
def home():
    return jsonify({
        "message": "Ingredient Expiry Prediction API is running!",
        "endpoints": ["/predict", "/health"],
        "model_loaded": model_loaded
    })

# ---------------------------
# Run server
# ---------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
