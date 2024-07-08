import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

# Load the saved model
model_hdf5_save_path = 'model.h5'
model = tf.keras.models.load_model(model_hdf5_save_path)

# Load dataset CSV for preprocessing information
file_path = 'dummy_data_with_bias.csv'
df = pd.read_csv(file_path)

# Preprocess dataset
df['Searches'] = df['Searches'].apply(lambda x: x.split(','))

# Use MultiLabelBinarizer to one-hot encode the product labels
mlb = MultiLabelBinarizer()
searches_encoded = mlb.fit_transform(df['Searches'])

# Combine encoded searches with original numerical features
X_numerical = df.drop(['Email', 'Searches'], axis=1).values.astype(np.float32)

# Standardize numerical features
scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X_numerical)

# Prepare new data for prediction
new_data = {
    'Email': ['c2tip58a@gmail.com'],
    'P01': [13],
    'P02': [3],
    'P03': [6],
    'P04': [14],
    'P05': [7],
    'P06': [20],
    'P07': [4],
    'P08': [18],
    'P09': [2],
    'P10': [13],
    'P11': [17],
    'P12': [11],
    'P13': [8],
    'P14': [3],
    'P15': [15]
}

new_df = pd.DataFrame(new_data)

# Drop the 'Email' column
new_df = new_df.drop(['Email'], axis=1)

# Convert to numpy array
new_X = new_df.values.astype(np.float32)

# Scale the data
new_X_scaled = scaler.transform(new_X)

# Create a zero array for one-hot encoded product labels
dummy_searches = np.zeros((new_X_scaled.shape[0], searches_encoded.shape[1]))

# Combine scaled numerical features with dummy one-hot encoded labels
new_X_combined = np.hstack((new_X_scaled, dummy_searches))

# Use the model to predict
predictions = model.predict(new_X_combined)

# Find the top 6 predictions
top_6_indices = np.argsort(predictions, axis=1)[0][-6:]
top_6_searches = [f'P{str(i+1).zfill(2)}' for i in top_6_indices]

print("Top 6 Predicted Searches:", top_6_searches)
