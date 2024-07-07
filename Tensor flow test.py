import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import TopKCategoricalAccuracy

# Load the dataset
df = pd.read_csv('dummy_data_with_bias.csv')

# Preprocess the data
# Create a DataFrame for searches
searches_df = df['Searches'].str.get_dummies(sep=',')
searches_columns = searches_df.columns

# Combine the original DataFrame with the searches DataFrame
df = df.drop('Searches', axis=1).join(searches_df.add_prefix('Search_'))

# Normalize the interaction features
interaction_columns = [col for col in df.columns if col.startswith('P')]
scaler = StandardScaler()
df[interaction_columns] = scaler.fit_transform(df[interaction_columns])

# Encode labels as binary
X = df.drop(['Email'] + [f'Search_{col}' for col in searches_columns], axis=1)
y = df[[f'Search_{col}' for col in searches_columns]]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=len(X_train.columns)))
model.add(Dropout(0.5))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=len(searches_columns), activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[TopKCategoricalAccuracy(k=6)])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Make predictions
y_hat = model.predict(X_test)
y_hat = (y_hat > 0.5).astype(int)

# Calculate accuracy for top 6 predictions
top_k_acc = TopKCategoricalAccuracy(k=6)
top_k_acc.update_state(y_test.astype('float32'), y_hat.astype('float32'))  # Convert to float32
accuracy = top_k_acc.result().numpy()

print(f"Top-6 Accuracy: {accuracy}")

# Save the model in h5 format
model.save('model.keras')

# Load the .h5 model
model = tf.keras.models.load_model('model.h5', custom_objects={'TopKCategoricalAccuracy': TopKCategoricalAccuracy})

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Set optimization
#tflite_model = converter.convert()

# (to generate a SavedModel) 
tf.saved_model.save(model, "saved_model_keras_dir")

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

'''
# Save the TensorFlow Lite model
with open('model.tflite', 'wb') as f:
    f.write(converter)
'''