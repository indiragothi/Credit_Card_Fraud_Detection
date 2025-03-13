from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib   
import os
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model using joblib instead of pickle
model_path = os.path.join('models', 'best_logistic_regression.pkl')
try:
    model = joblib.load(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Initialize the scaler for standardizing Amount
scaler = StandardScaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form - only V1-V28 features and Amount
        features = []
        for i in range(1, 29):  # V1-V28
            feature_name = f'V{i}'
            feature_value = float(request.form.get(feature_name, 0))
            features.append(feature_value)
        
        # Get Amount and standardize it
        amount = float(request.form.get('Amount', 0))
        sample_amounts = np.array([0, 1, 10, 100, 1000, 10000])  # Sample range
        scaler.fit(sample_amounts.reshape(-1, 1))
        std_amount = scaler.transform([[amount]])[0][0]
        
        # Add standardized amount to features
        features.append(std_amount)
        
        # Convert to numpy array and reshape for prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)
        probability = model.predict_proba(features_array)
        
        # Format result
        fraud_probability = round(probability[0][1] * 100, 2)
        result = "Fraudulent" if prediction[0] == 1 else "Legitimate"
        
        return jsonify({
            'prediction': result,
            'probability': fraud_probability,
            'status': 'success'
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()  # Print detailed error for debugging
        return jsonify({
            'prediction': 'Error',
            'probability': 0,
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)