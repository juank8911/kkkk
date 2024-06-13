# j:\ProyectosCriptoMon\UPGPR\main.py
# UPGPR/main.py

import tensorflow as tf
import sys
import os
import threading
import dotenv
from src.aitrini.base.Trinity import create_model  # Importar la función create_model
from src.aitrini.treaning.TreanModel import TreanModel  # Importar la clase TreanModel

dotenv.load_dotenv()  # Cargar variables de entorno desde .env

# Obtener las claves API y secreta desde el archivo .env
api_key = os.getenv("api_key")
secret_key = os.getenv("secret_key")

# Definir la forma de entrada del modelo
input_shape = (10,)  # Ejemplo: 10 características

# Definir la ruta al archivo de resultados de entrenamiento
training_results_file = "src/aitrini/tets/treaninData/training_results.txt"

# Función para iniciar el entrenamiento asíncrono
def iniciar_entrenamiento():
    # Verificar si el archivo de resultados de entrenamiento existe
    if os.path.exists(training_results_file):
        # Cargar los resultados de entrenamiento
        with open(training_results_file, "r") as f:
            accuracy = float(f.readline().strip().split(":")[1])

        # Verificar si la precisión es mayor o igual al 80%
        if accuracy >= 0.8:
            print("Modelo Trinity ya entrenado con una precisión del 80% o más.")
            # Cargar el modelo entrenado
            model = tf.keras.models.load_model("src/aitrini/tets/treaninData/Trinity.keras")
        else:
            print("Modelo Trinity necesita entrenamiento.")
            # Crear y compilar el modelo
            model = create_model(input_shape)
            model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

            # Entrenar el modelo
            trean_model = TreanModel(api_key=api_key, secret_key=secret_key, tfrecords_path="src/aitrini/tets/treaninData/ohlcv_data.tfrecords", input_shape=input_shape, target_shape=(1,))
            trean_model.run()

            # Guardar el modelo entrenado
            model.save("src/aitrini/tets/treaninData/Trinity.keras")

            # Guardar los resultados de entrenamiento
            with open(training_results_file, "w") as f:
                f.write(f"Accuracy: {accuracy}")

    else:
        print("Modelo Trinity necesita entrenamiento.")
        # Crear y compilar el modelo
        model = create_model(input_shape)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # Entrenar el modelo
        trean_model = TreanModel(api_key=api_key, secret_key=secret_key, tfrecords_path="src/aitrini/tets/treaninData/ohlcv_data.tfrecords", input_shape=input_shape, target_shape=(1,))
        trean_model.run()

        # Guardar el modelo entrenado
        model.save("src/aitrini/tets/treaninData/Trinity.keras")

        # Guardar los resultados de entrenamiento
        with open(training_results_file, "w") as f:
            f.write(f"Accuracy: {accuracy}")

# Iniciar el entrenamiento en un hilo separado
training_thread = threading.Thread(target=iniciar_entrenamiento)
training_thread.start()

# Espacio de despliegue (por ahora comentado)
# # ... (código de despliegue) ...

print("Bienvenido, soy Trinity.")