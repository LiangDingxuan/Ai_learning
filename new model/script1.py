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
model.export(model_save_path)

# Convert the saved model to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(model_save_path)
tflite_model = converter.convert()

# Save the TFLite model
tflite_model_path = 'model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f'TFLite model saved to {tflite_model_path}')
