import os
print("Model file exists:", os.path.exists('models/final_iris_kNN_pipeline.pkl'))

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the pre-trained pipeline model
model = joblib.load('models/final_iris_kNN_pipeline.pkl')

# Dictionary to map class labels to species names
species_dict = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

@app.route('/')
def home():
    return '''
    <html>
        <head>
            <title>Iris Predictor</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 600px; margin: auto; }
                .form-group { margin-bottom: 15px; }
                input { width: 100%; padding: 8px; margin-top: 5px; }
                button { padding: 10px 20px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
                button:hover { background-color: #45a049; }
                #result { margin-top: 20px; padding: 10px; }
                .error { color: red; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Iris Flower Predictor</h1>
                <form id="prediction-form">
                    <div class="form-group">
                        <label>Sepal Length:</label>
                        <input type="number" step="0.1" name="sepal_length" required>
                    </div>
                    <div class="form-group">
                        <label>Sepal Width:</label>
                        <input type="number" step="0.1" name="sepal_width" required>
                    </div>
                    <div class="form-group">
                        <label>Petal Length:</label>
                        <input type="number" step="0.1" name="petal_length" required>
                    </div>
                    <div class="form-group">
                        <label>Petal Width:</label>
                        <input type="number" step="0.1" name="petal_width" required>
                    </div>
                    <button type="submit">Predict</button>
                </form>
                <div id="result"></div>
            </div>
            
            <script>
                document.getElementById('prediction-form').addEventListener('submit', async (e) => {
                    e.preventDefault();
                    const resultDiv = document.getElementById('result');
                    resultDiv.innerHTML = 'Processing...';
                    
                    const formData = new FormData(e.target);
                    const features = [
                        parseFloat(formData.get('sepal_length')),
                        parseFloat(formData.get('sepal_width')),
                        parseFloat(formData.get('petal_length')),
                        parseFloat(formData.get('petal_width'))
                    ];
                    
                    try {
                        console.log('Sending features:', features); // Debug log
                        
                        const response = await fetch('/predict', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({features: [features]}),
                        });
                        
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        
                        const data = await response.json();
                        console.log('Received response:', data); // Debug log
                        
                        if (data.error) {
                            throw new Error(data.error);
                        }
                        
                        if (data.prediction) {
                            resultDiv.innerHTML = `<h3>Predicted Species: ${data.prediction}</h3>`;
                        } else {
                            throw new Error('No prediction in response');
                        }
                    } catch (error) {
                        console.error('Error:', error); // Debug log
                        resultDiv.innerHTML = `<p class="error">Error: ${error.message}</p>`;
                    }
                });
            </script>
        </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'No features provided'}), 400
            
        # Extract the features from the input data
        input_data = np.array(data['features'])
        
        # Add debug logging
        print("Received input data:", input_data)
        
        features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

        # Convert input data into a pandas DataFrame for consistency
        input_df = pd.DataFrame(input_data, columns=features)
        
        # Add debug logging
        print("Created DataFrame:", input_df)
        
        # Predict using the pipeline
        prediction = model.predict(input_df)
        
        # Add debug logging
        print("Model prediction:", prediction)
        
        # Map the prediction to species name
        species_name = species_dict[prediction[0]]
        
        # Add debug logging
        print("Species name:", species_name)
        
        # Return the species name as JSON response
        return jsonify({'prediction': species_name})
    except Exception as e:
        print("Error occurred:", str(e))  # Debug logging
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)