from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the pre-trained pipeline model
model = joblib.load('models/final_iris_SVC_pipeline.pkl')

# Dictionary to map class labels to species names
species_dict = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.get_json()  # Expects data to be in JSON format
    
    # Extract the features from the input data
    input_data = np.array(data['features'])  # 'features' should be a list in the JSON
    
    # Convert input data into a pandas DataFrame for consistency
    input_df = pd.DataFrame(input_data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    
    # Predict using the pipeline
    prediction = model.predict(input_df)
    
    # Map the prediction to species name
    species_name = species_dict[prediction[0]]
    
    # Return the species name as JSON response
    return jsonify({'prediction': species_name})

# Run the Flask app (make sure to use gunicorn in production)
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
