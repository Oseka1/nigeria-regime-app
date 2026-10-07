from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your saved model from the NBE python script step
model = joblib.load('regime_shift_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        raw_features = data['features']

        # Converts all inputs (Year, Political_Era encoding, Pct metrics, Oil Price) into floats
        features = [float(x) for x in raw_features]

        # Structure into a 2D array matrix: [[Year, Era, Agric, Industry, Services, Oil]]
        final_features = [np.array(features)]

        # Predict the numerical Official_FX_Rate_USD_NGN output
        prediction = model.predict(final_features)

        # Clean array formatting and bound to 2 decimal places for Naira presentation
        fx_rate_pred = float(prediction[0])
        formatted_prediction = f"{fx_rate_pred:,.2f}"

        return jsonify({'prediction': formatted_prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True) 