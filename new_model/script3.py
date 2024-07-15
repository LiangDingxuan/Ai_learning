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

save_path = 'model.h5'
accuracy_threshold = 0.90

# Loop to train model until accuracy exceeds threshold
accuracy = 0.0
while accuracy <= accuracy_threshold:
    model = build_model(input_shape=X_train.shape[1], output_shape=15, learning_rate=0.001, l1=1e-5, l2=1e-4, dropout_rate=0.1)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train model
    history = model.fit(X_train, y_train, epochs=60, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

    if accuracy > accuracy_threshold:
        print(f"\nAccuracy threshold of {accuracy_threshold * 100}% exceeded. Saving model...................Saving model")
        model.save(save_path)
        print(f'Keras model saved to {save_path}')
        
        # Save to saved model file
        model_save_path = 'saved_model'
        tf.saved_model.save(model, model_save_path)
        
        # Convert the saved model to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(model_save_path)
        tflite_model = converter.convert()
        
        # Save the TFLite model
        tflite_model_path = 'model.tflite'
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        print(f'TFLite model saved to {tflite_model_path}')
        break

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

# Use the model to predict
predictions = model.predict(new_X_scaled)

# Get the predicted sequence of indices for each product
predicted_indices = [np.argsort(pred)[::-1] for pred in predictions[0]]
predicted_searches = [[f'P{str(idx+1).zfill(2)}' for idx in indices] for indices in predicted_indices]

print("Predicted Searches in Order:", predicted_searches[0])
