import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import pandas as pd

# Define the data
data = {
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
    'P15': [7],
    'Searches': ["P01,P03,P05,P06,P12,P10"]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Preprocess the dataset
df['Searches'] = df['Searches'].apply(lambda x: x.split(','))

# Use MultiLabelBinarizer to one-hot encode the product labels
mlb = MultiLabelBinarizer()
searches_encoded = mlb.fit_transform(df['Searches'])

# Extract and standardize numerical features
X_numerical = df.drop(['Email', 'Searches'], axis=1).values.astype(np.float32)
scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X_numerical)

# Ensure consistent preprocessing by loading the training data and fitting the scaler and binarizer
training_data = pd.read_csv('dummy_data_with_bias.csv')
training_data['Searches'] = training_data['Searches'].apply(lambda x: x.split(','))
training_searches_encoded = mlb.fit_transform(training_data['Searches'])
training_X_numerical = training_data.drop(['Email', 'Searches'], axis=1).values.astype(np.float32)
scaler.fit(training_X_numerical)

# Transform the test data
X_numerical_scaled = scaler.transform(X_numerical)
X_test = np.hstack((X_numerical_scaled, searches_encoded)).astype(np.float32)

# Load the TFLite model and allocate tensors
tflite_model_path = 'model.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Check the expected input shape
expected_input_shape = input_details[0]['shape']
print(f'Expected input shape: {expected_input_shape}')

# Set the input tensor, ensure the shape matches the expected shape
print(f"Current shape of X_test: {X_test.shape}")
# Expected shape
expected_features = 30

# If X_test does not have the expected shape, you need to adjust it.
# This could involve adding missing features, or modifying the preprocessing pipeline.
# This is a placeholder for whatever steps are necessary to adjust X_test.
# For example, if missing features are zeros:
if X_test.shape[1] < expected_features:
    missing_features = expected_features - X_test.shape[1]
    X_test = np.hstack([X_test, np.zeros((X_test.shape[0], missing_features))])

# Validate the shape after adjustment
assert X_test.shape[1] == expected_features, f"Shape after adjustment: {X_test.shape[1]}, expected: {expected_features}"

if X_test.shape[1] != expected_input_shape[1]:
    raise ValueError(f"Dimension mismatch: Got {X_test.shape[1]} but expected {expected_input_shape[1]}")

# Add a batch dimension to the test data if necessary
if len(X_test.shape) == 1:
    X_test = np.expand_dims(X_test, axis=0)

# Convert X_test to FLOAT32 before setting the tensor
X_test = X_test.astype(np.float32)
# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], X_test)

# Invoke the model
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Convert predictions to binary outputs
predictions_binary = (output_data > 0.5).astype(int)

# Decode the predictions to product labels
predicted_products = mlb.inverse_transform(predictions_binary)
print(f'Predicted products: {predicted_products}')

# Compare with actual labels
actual_labels = searches_encoded

import numpy as np

# Assuming actual_labels is a numpy array with shape (n_samples, 6)
# And you have a MultiLabelBinarizer (mlb) trained on 15 classes

# Create an empty array with the correct shape (n_samples, 15)
adjusted_labels = np.zeros((actual_labels.shape[0], 15))

# Assuming the 6 classes in actual_labels correspond to the first 6 classes the mlb was trained on
# Copy the actual_labels into the first 6 columns of adjusted_labels
adjusted_labels[:, :6] = actual_labels

# Now use the adjusted_labels with the correct shape for inverse_transform
actual_products = mlb.inverse_transform(adjusted_labels)

print(f'Actual products: {actual_products}')
