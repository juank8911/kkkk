# UPGPR/main.py

import tensorflow as tf
import sys
import os

try:
    # Import the Trinity model
    from src.aitrini.base.Trinity import model

    # Load the saved model
    model = tf.keras.models.load_model('Trinity.h5')

    # Example usage:
    # Assuming you have input data in a NumPy array called 'input_data'
    # predictions = model.predict(input_data)

    # Process the predictions as needed
    # print(predictions)

    print("Bienvenido, soy Trinity.")

except Exception as e:
    print(f"Error al iniciar: {e}")
