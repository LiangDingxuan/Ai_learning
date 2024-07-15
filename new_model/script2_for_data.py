import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import l1_l2
import csv

# Prepare dataset--------------------------------------------------------------------------
# Load dataset CSV
file_path = 'generated_data.csv'
df = pd.read_csv(file_path)

# Preprocess dataset
df['Searches'] = df['Searches'].apply(lambda x: x.split(','))

# Use a mapping to convert product labels to indices
product_mapping = {f'P{str(i+1).zfill(2)}': i for i in range(15)}
df['Searches'] = df['Searches'].apply(lambda x: [product_mapping[item] for item in x])

# Combine encoded searches with original numerical features
X_numerical = df.drop(['Email', 'Searches'], axis=1).values.astype(np.float32)

# Standardize numerical features
scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X_numerical)

# Prepare target labels (encoded searches)
y = np.array(df['Searches'].tolist()).astype(np.float32)
y = np.expand_dims(y, axis=-1)  # Ensure the target has shape (num_samples, 15, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_numerical_scaled, y, test_size=0.2, random_state=42)

# Train model------------------------------------------------------------------------------------------
def build_model(input_shape, output_shape, learning_rate=0.001, l1=1e-5, l2=1e-4, dropout_rate=0.3):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,), kernel_regularizer=l1_l2(l1=l1, l2=l2)),
        BatchNormalization(),
        Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=l1, l2=l2)),
        Dropout(dropout_rate),
        BatchNormalization(),
        Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=l1, l2=l2)),
        Dropout(dropout_rate),
        Dense(output_shape * 15, activation='linear'),  # Output shape is 15 * 15 classes
        Reshape((15, output_shape))
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=SparseCategoricalCrossentropy(from_logits=True), 
                  metrics=['accuracy'])
    return model

results = []
dropout_rates = np.arange(0.1, 0.31, 0.01)

for dropout_rate in dropout_rates:
    for i in range(5):
        print("Dropout rate ======================================== " , dropout_rate)
        model = build_model(input_shape=X_train.shape[1], output_shape=15, learning_rate=0.001, l1=1e-5, l2=1e-4, dropout_rate=dropout_rate)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train model
        history = model.fit(X_train, y_train, epochs=60, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)

        # Evaluate model
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        results.append((dropout_rate, i + 1, loss, accuracy))
        print(f'Dropout Rate: {dropout_rate}, Run: {i + 1}, Loss: {loss}, Accuracy: {accuracy}')

# Write results to CSV file
results_df = pd.DataFrame(results, columns=['Dropout Rate', 'Run', 'Loss', 'Accuracy'])
results_df.to_csv('dropout_results.csv', index=False)
print('Results saved to dropout_results.csv')
