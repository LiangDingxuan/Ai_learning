import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Prepare new data for prediction
new_data = {
    'Email': ['c2tip58a@gmail.com'],
    'P01': [19],
    'P02': [0],
    'P03': [19],
    'P04': [3],
    'P05': [17],
    'P06': [12],
    'P07': [6],
    'P08': [2],
    'P09': [0],
    'P10': [8],
    'P11': [7],
    'P12': [10],
    'P13': [4],
    'P14': [0],
    'P15': [7]
}

new_df = pd.DataFrame(new_data)

# Drop the 'Email' column
new_df = new_df.drop(['Email'], axis=1)

# Convert to numpy array
new_X = new_df.values.astype(np.float32)

# Scale the data
new_X_scaled = scaler.transform(new_X)

# Create a zero array for one-hot encoded product labels
dummy_searches = np.zeros((new_X_scaled.shape[0], len(input_details[0]['shape']) - new_X_scaled.shape[1]))

# Combine scaled numerical features with dummy one-hot encoded labels
new_X_combined = np.hstack((new_X_scaled, dummy_searches))

# Set the tensor to point to the input data to be inferred
interpreter.set_tensor(input_details[0]['index'], new_X_combined)

# Run the interpreter
interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data
# Use `tensor()` in order to get a pointer to the tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Find the top 6 predictions
top_6_indices = np.argsort(output_data, axis=1)[0][-6:]
top_6_searches = [f'P{str(i+1).zfill(2)}' for i in top_6_indices]

print("Top 6 Predicted Searches:", top_6_searches)
