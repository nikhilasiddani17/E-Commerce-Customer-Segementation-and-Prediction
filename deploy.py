from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

# Create Flask application
app = Flask(__name__)

# Load the model and scaler
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML form

@app.route('/predict', methods=['POST'])
def predict():
    # Check if JSON data is present in the request
    if not request.json:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        # Load JSON data into a DataFrame
        data = request.json
        # Ensure data is wrapped in a list to create a DataFrame
        df = pd.DataFrame([data])

        # Verify the required columns are present
        required_columns = ['Recency', 'Frequency', 'Monetary']
        if not all(column in df.columns for column in required_columns):
            return jsonify({'error': f'Missing columns in input data. Required columns: {required_columns}'}), 400

        # Preprocess the data
        scaled_data = scaler.transform(df[required_columns])
        predictions = kmeans.predict(scaled_data)

        # Return predictions as a JSON response
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
