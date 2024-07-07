import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

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

'''
# Build model
model = tf.keras.Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    BatchNormalization(),
    Dense(256, activation='relu'),
    Dropout(0.3),  # Adjusted dropout rate
    BatchNormalization(),
    Dense(128, activation='relu'),
    Dropout(0.3),  # Adjusted dropout rate
    Dense(y.shape[1], activation='sigmoid')
]) 

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Experiment with learning rates
              loss='binary_crossentropy',  # Use binary_crossentropy for multi-label classification
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
'''
#'''
def build_model(input_shape, output_shape, learning_rate=0.001, l1=1e-5, l2=1e-4, dropout_rate=0.23):# 0.25 > dropout rate > 0.2 best results currently: 0.23
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,), kernel_regularizer=l1_l2(l1=l1, l2=l2)),
        BatchNormalization(),
        Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=l1, l2=l2)),
        Dropout(dropout_rate),
        BatchNormalization(),
        Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=l1, l2=l2)),
        Dropout(dropout_rate),
        Dense(output_shape, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model

# Example of using the function to create a model
model = build_model(input_shape=X_train.shape[1], output_shape=y_train.shape[1], learning_rate=0.001, l1=1e-5, l2=1e-4, dropout_rate=0.3)

#'''


early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train model
history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test))  # Reduce epochs to avoid overfitting

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
dummy_searches = np.zeros((new_X_scaled.shape[0], searches_encoded.shape[1]))

# Combine scaled numerical features with dummy one-hot encoded labels
new_X_combined = np.hstack((new_X_scaled, dummy_searches))

# Use the model to predict
predictions = model.predict(new_X_combined)

# Find the top 6 predictions
top_6_indices = np.argsort(predictions, axis=1)[0][-6:]
top_6_searches = [f'P{str(i+1).zfill(2)}' for i in top_6_indices]

print("Top 6 Predicted Searches:", top_6_searches)
