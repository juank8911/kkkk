# UPGPR/src/aitrini/base/Trinity.py

import tensorflow as tf

# Define the model
def create_model(input_shape):
    trinity = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),  # Capa 1: 128 neuronas, activación ReLU
        tf.keras.layers.Dense(68, activation='relu'),                        # Capa 2: 68 neuronas, activación ReLU
        tf.keras.layers.Dense(1, activation='sigmoid')                       # Capa 3: 1 neurona, activación Sigmoid
    ])
    return trinity

# Define the input shape (replace with your actual input shape)
input_shape = (10,)  # Example: 10 features

# Compile the model
trinity = create_model(input_shape)
trinity.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Save the model
trinity.save('Trinity.h5')

print("Modelo Trinity creado y guardado en Trinity.h5")
