import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Load dataset CSV
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

X = np.hstack((X_numerical_scaled, searches_encoded))

# Prepare target labels (encoded searches)
y = searches_encoded.astype(np.float32)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Add dropout to prevent overfitting
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(y.shape[1], activation='sigmoid')  # Adjust output layer for multi-label classification
])

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Experiment with learning rates
              loss='binary_crossentropy',  # Use binary_crossentropy for multi-label classification
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Train model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))  # Reduce epochs to avoid overfitting

# Evaluate model
loss, accuracy, precision, recall = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}')

# Predict using model
predictions = model.predict(X_test)

# Convert predictions to binary outputs
predictions_binary = (predictions > 0.5).astype(int)

# Evaluate model using F1-score
f1 = f1_score(y_test, predictions_binary, average='weighted')
print(f'F1 Score: {f1}')

# Save the Keras model
model_save_path = 'saved_model'
model.export(model_save_path)  # Corrected from model.export to model.save

# Convert the saved model to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(model_save_path)
tflite_model = converter.convert()

# Save the TFLite model
tflite_model_path = 'model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f'TFLite model saved to {tflite_model_path}')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler  # Assuming you're using StandardScaler for scaling

# Assuming `model` is your trained model and `scaler` is your StandardScaler instance

# Updated new_data without 'Searches'
new_data = {
    'Email': ['c2tip58a@gmail.com'],  # This line is not needed for prediction and will be removed in preprocessing
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
# Assuming `model` is your TensorFlow/Keras model
input_shape = model.input_shape
print("Model's expected input shape:", input_shape)

# If the input shape is not [1,15], adjust your model architecture or data preprocessing accordingly
# Convert to DataFrame
new_df = pd.DataFrame(new_data)

# Drop 'Email' column as it's not used in prediction
new_df = new_df.drop(['Email'], axis=1)

# Convert to numpy array
new_X = new_df.values.astype(np.float32)

# Scale the data
# Assuming `scaler` has been fitted on the training data
new_X_scaled = scaler.transform(new_X)

# Use the model to predict
predictions = model.predict(new_X_scaled)

# Assuming the model outputs probabilities for each class (P01 to P15),
# and you need to find the top 6 predictions
top_6_indices = np.argsort(predictions, axis=1)[0][-6:]
top_6_searches = new_df.columns[top_6_indices].tolist()

print("Top 6 Predicted Searches:", top_6_searches)