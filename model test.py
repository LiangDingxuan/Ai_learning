import tensorflow as tf
'''
# Load the model
model = tf.keras.models.load_model('saved_model.keras')
# Assuming X_test is your test data
y_pred = model.predict(X_test)
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Example data
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

# Create DataFrame
df = pd.DataFrame(data)

# Preprocess the data
# Split the 'Searches' into separate columns
searches_df = df['Searches'].str.get_dummies(sep=',')
searches_columns = searches_df.columns

# Combine the original DataFrame with the searches DataFrame
df = df.drop('Searches', axis=1).join(searches_df.add_prefix('Search_'))

# Normalize the interaction features
interaction_columns = [col for col in df.columns if col.startswith('P')]
scaler = StandardScaler()
df[interaction_columns] = scaler.fit_transform(df[interaction_columns])

# Remove 'Email' column as it's not used for prediction
X_test = df.drop(['Email'] + [f'Search_{col}' for col in searches_columns], axis=1)

# Load the model
import tensorflow as tf
model = tf.keras.models.load_model('saved_model.keras')

# Make predictions
y_pred = model.predict(X_test)

# Example: Print the predicted probabilities for the first example
print("Predicted probabilities:")
print(y_pred[0])

# Example: Convert probabilities to predicted classes (assuming binary classification)
threshold = 0.5
y_pred_binary = (y_pred > threshold).astype(int)
print("Predicted classes:")
print(y_pred_binary[0])
