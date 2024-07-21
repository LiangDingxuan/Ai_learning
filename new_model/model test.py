# testing.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load scaler
file_path = 'generated_data.csv'
df = pd.read_csv(file_path)
X_numerical = df.drop(['Email', 'Searches'], axis=1).values.astype(np.float32)
scaler = StandardScaler()
scaler.fit(X_numerical)

# Load the trained model
model_path = 'model.h5'
model = tf.keras.models.load_model(model_path)

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
new_df = new_df.drop(['Email'], axis=1)
new_X = new_df.values.astype(np.float32)
new_X_scaled = scaler.transform(new_X)
predictions = model.predict(new_X_scaled)

predicted_indices = [np.argsort(pred)[::-1] for pred in predictions[0]]
predicted_searches = [[f'P{str(idx+1).zfill(2)}' for idx in indices] for indices in predicted_indices]

print("Predicted Searches in Order:", predicted_searches[0])
