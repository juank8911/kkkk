# src/aitrini/base/Trinity.py
import tensorflow as tf
import os
import dotenv

# Load environment variables from .env
dotenv.load_dotenv()

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

# Check if the training results file exists and if the accuracy is above 80%
training_results_file = "src/aitrini/tets/treaninData/training_results.txt"
if os.path.exists(training_results_file):
    with open(training_results_file, "r") as f:
        accuracy = float(f.readline().strip().split(":")[1])
        if accuracy >= 0.8:
            print("Modelo Trinity ya entrenado con una precisión del 80% o más.")
            # Load the trained model
            model = tf.keras.models.load_model("src/aitrini/tets/treaninData/Trinity.keras")
        else:
            print("Modelo Trinity necesita entrenamiento.")
            # Create and compile the model
            model = create_model(input_shape)
            model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
            # Train the model (call the training function from TreanModel.py)
            from src.aitrini.treaning.TreanModel import TreanModel
            api_key = os.getenv("API_KEY")
            secret_key = os.getenv("SECRET_KEY")
            trean_model = TreanModel(api_key=api_key, secret_key=secret_key, tfrecords_path="src/aitrini/tets/treaninData/ohlcv_data.tfrecords", input_shape=input_shape, target_shape=(1,))
            trean_model.train()
            # Save the trained model
            model.save("src/aitrini/tets/treaninData/Trinity.keras")
            # Save the training results
            with open(training_results_file, "w") as f:
                f.write(f"Accuracy: {accuracy}")
else:
    print("Modelo Trinity necesita entrenamiento.")
    # Create and compile the model
    model = create_model(input_shape)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # Train the model (call the training function from TreanModel.py)
    from src.aitrini.treaning.TreanModel import TreanModel
    api_key = os.getenv("api_key")
    secret_key = os.getenv("secret_key")
    trean_model = TreanModel(api_key=api_key, secret_key=secret_key, tfrecords_path="src/aitrini/tets/treaninData/ohlcv_data.tfrecords", input_shape=input_shape, target_shape=(1,))
    trean_model.train()
    # Save the trained model
    model.save("src/aitrini/tets/treaninData/Trinity.keras")
    # Save the training results
    with open(training_results_file, "w") as f:
        f.write(f"Accuracy: {accuracy}")

print("Modelo Trinity creado y guardado en Trinity.keras")
