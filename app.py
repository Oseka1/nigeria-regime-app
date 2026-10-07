from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model, robust scaler, and label encoder
model = joblib.load('regime_shift_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        raw_features = data['features']

        # 1. Extract inputs cleanly
        year = float(raw_features[0])
        political_era_code = float(raw_features[1]) # Dropdown sends encoded numeric integer directly
        agric = float(raw_features[2])
        industry = float(raw_features[3])
        services = float(raw_features[4])
        oil_price = float(raw_features[5])

        # 2. Structure columns into a DataFrame matching your exact original X layout order
        # IMPORTANT: Ensure these match your original training column names perfectly
        input_data = pd.DataFrame([{
            'Year': year,
            'Political_Era': political_era_code,
            'Agriculture_Contribution_Pct': agric,
            'Industry_Contribution_Pct': industry,
            'Services_Contribution_Pct': services,
            'Average_Crude_Oil_Price_USD': oil_price
        }])

        # 3. Apply RobustScaler transformation to features (excluding the encoded categorical if relevant)
        # Note: If your original script scaled the entire X matrix together, keep it like this:
        scaled_features = scaler.transform(input_data)
        
        # 4. Predict the numerical Official_FX_Rate_USD_NGN output
        prediction = model.predict(scaled_features)

        # Format the continuous value to 2 decimal places for presentation
        fx_rate_pred = float(prediction)
        formatted_prediction = f"{fx_rate_pred:,.2f}"

        return jsonify({'prediction': formatted_prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True) 
