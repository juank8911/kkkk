# src/aitrini/treaning/TreanModel.py
import tensorflow as tf
import pandas as pd
from src.aitrini.base.Trinity import create_model  # Importar la función create_model
from ..api.BinanceAdp import BinanceAdp  # Importar la clase BinanceAdp
import time
import ccxt
import os
import dotenv
import threading
import numpy as np

dotenv.load_dotenv()

class TreanModel:
    def __init__(self, api_key, secret_key, input_shape, target_shape, learning_rate=0.001):
        """
        Inicializa el modelo de entrenamiento.

        Args:
            api_key (str): La clave API de Binance.
            secret_key (str): La clave secreta de Binance.
            input_shape (tuple): La forma de los datos de entrada.
            target_shape (tuple): La forma de los datos de salida.
            learning_rate (float): La tasa de aprendizaje del optimizador.
        """
        self.api_key = api_key  #llamar apiKey dese .env
        self.secret_key = secret_key
        self.input_shape = input_shape
        self.target_shape = target_shape
        self.learning_rate = learning_rate

        # Crear la instancia de BinanceAdp
        self.binance_adapter = BinanceAdp(api_key, secret_key)

        # Crear el modelo Trinity
        self.model = create_model(input_shape)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                         loss='binary_crossentropy',
                         metrics=['accuracy'])

        # Variables para seguimiento de ganancias
        self.total_profit = 0
        self.total_trades = 0
        self.accuracy = 0

        # Variable para controlar el entrenamiento
        self.training_stopped = False

    def train(self, epochs=10, batch_size=32):
        """
        Entrena el modelo.

        Args:
            epochs (int): El número de épocas de entrenamiento.
            batch_size (int): El tamaño del lote de entrenamiento.
        """
        # Obtener datos históricos de BinanceAdp
        tfrecords_path = self.binance_adapter.get_historical_futures_data("15s", 10800, 1000)  # 3 horas en segundos (10800), velas de 15 segundos ("15s") y 1000 velas

        # Cargar los datos de entrenamiento desde TFRecords
        self.train_dataset = tf.data.TFRecordDataset(tfrecords_path).map(self.parse_tfrecord)

        # Reshape the data to match the model's expected input shape
        self.train_dataset = self.train_dataset.map(lambda x, y: (tf.reshape(x, (1, -1)), y))

        history = self.model.fit(self.train_dataset, epochs=epochs, batch_size=batch_size)
        self.accuracy = history.history['accuracy'][-1]
        if self.accuracy >= 0.8:
            self.training_stopped = True
            print("Entrenamiento detenido. Se alcanzó una precisión del 80% o más.")

    def predict(self, data):
        """
        Realiza una predicción con el modelo.

        Args:
            data (tf.Tensor): Los datos de entrada.

        Returns:
            tf.Tensor: La predicción del modelo.
        """
        return self.model.predict(data)

    def trade(self, symbol, timeframe, minutes_ago, leverage):
        """
        Realiza una operación de compra y venta en Binance.

        Args:
            symbol (str): El símbolo del contrato de futuros.
            timeframe (str): El intervalo de tiempo para los datos históricos.
            minutes_ago (int): El número de minutos para retroceder en el tiempo para los datos históricos.
            leverage (int): El apalancamiento a utilizar.
        """
        # Obtener datos históricos
        historical_data = self.binance_adapter.get_historical_futures_data(timeframe, minutes_ago, 1000)  # 3 horas en segundos (10800), velas de 15 segundos ("15s") y 1000 velas

        # Convertir los datos a un DataFrame de Pandas
        df = pd.DataFrame(historical_data["data"])
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]

        # Realizar una predicción con el modelo
        prediction = self.predict(df["close"].values.reshape(1, -1))

        # Simular la compra o venta
        self.simulate_trade(symbol, prediction, leverage)

    def simulate_trade(self, symbol, prediction, leverage):
        """
        Simulates a buy or sell trade with dynamic profit targets and time limits.

        Args:
            symbol (str): The futures contract symbol.
            prediction (float): The model's prediction.
            leverage (int): The leverage to use.
        """

        self.total_trades += 1
        current_price = self.binance_adapter.exchange.fetch_ticker(symbol)["last"]

        if prediction > 0.5:  # Buy trade
            # Minimum profit target and time limits
            min_profit_per_30_seconds = 0.0015 * leverage
            max_trade_duration_seconds = 300  # 5 minutes

            # Calculate initial profit target
            target_price = current_price * (1 + min_profit_per_30_seconds)

            # Simulate trade with dynamic profit target and time limit
            profit = 0
            start_time = time.time()
            while profit < min_profit_per_30_seconds and (time.time() - start_time) < max_trade_duration_seconds:
                # Check current price and update profit
                new_price = self.binance_adapter.exchange.fetch_ticker(symbol)["last"]
                profit = (new_price - current_price) * leverage

                # Adjust profit target if necessary
                elapsed_time_seconds = time.time() - start_time
                if elapsed_time_seconds % 30 == 0:
                    min_profit_per_30_seconds += 0.0015 * leverage
                    target_price = current_price * (1 + min_profit_per_30_seconds)

                # Sleep for 1 second
                time.sleep(1)

            # Print trade summary
            print(f"Compra simulada de {symbol} con un apalancamiento de {leverage}. Ganancia: {profit:.2f}")

        else:  # Sell trade
            # Minimum profit target and time limits
            min_profit_per_30_seconds = 0.0015 * leverage
            max_trade_duration_seconds = 300  # 5 minutes

            # Calculate initial profit target
            target_price = current_price * (1 - min_profit_per_30_seconds)

            # Simulate trade with dynamic profit target and time limit
            profit = 0
            start_time = time.time()
            while profit < min_profit_per_30_seconds and (time.time() - start_time) < max_trade_duration_seconds:
                # Check current price and update profit
                new_price = self.binance_adapter.exchange.fetch_ticker(symbol)["last"]
                profit = (current_price - new_price) * leverage

                # Adjust profit target if necessary
                elapsed_time_seconds = time.time() - start_time
                if elapsed_time_seconds % 30 == 0:
                    min_profit_per_30_seconds += 0.0015 * leverage
                    target_price = current_price * (1 - min_profit_per_30_seconds)

                # Sleep for 1 second
                time.sleep(1)

            # Print trade summary
            print(f"Venta simulada de {symbol} con un apalancamiento de {leverage}. Ganancia: {profit:.2f}")

    def close_position(self, symbol):
        """
        Cierra la posición actual.

        Args:
            symbol (str): El símbolo del contrato de futuros.
        """
        # Obtener la posición actual
        position = self.binance_adapter.exchange.fetch_positions(symbol)[0]

        # Cerrar la posición
        self.binance_adapter.exchange.create_market_order(symbol, position["amount"], "sell" if position["amount"] > 0 else "buy")

    def get_optimal_leverage(self, symbol):
        """
        Calcula el apalancamiento óptimo para un símbolo dado en base al modelo entrenado.

        Args:
            symbol (str): El símbolo del contrato de futuros.

        Returns:
            float: El apalancamiento óptimo predicho (puede no ser un número entero).

        Raises:
            ValueError: Si el modelo aún no ha sido entrenado.
        """

        if not self.model:
            raise ValueError("El modelo no ha sido entrenado. Llame a 'train' antes de usar 'get_optimal_leverage'.")

        # Prepare los datos de precios históricos para el símbolo especificado (reemplace con su lógica de carga de datos)
        features = self.load_symbol_data(symbol)  # Detalles de implementación aquí

        # Asegúrese de que las características estén en el formato esperado
        features = np.expand_dims(features, axis=0)  # Agregue la dimensión de lote si es necesario

        # Prediga el apalancamiento óptimo utilizando el modelo entrenado
        predicted_leverage = self.model.predict(features)[0][0]

        # Valide o ajuste la predicción (opcional)
        # Este paso podría implicar un filtrado basado en el conocimiento del dominio o la tolerancia al riesgo.

        return predicted_leverage

    def load_symbol_data(self, symbol):
        """
        Carga los datos de precios históricos para un símbolo dado.

        Args:
            symbol (str): El símbolo del contrato de futuros.

        Returns:
            np.ndarray: Un array NumPy que contiene los datos de precios históricos.
        """
        # Reemplace esta lógica con su implementación real de carga de datos
        # Por ejemplo, podría usar self.binance_adapter.get_historical_futures_data
        # para obtener los datos y luego procesarlos en un array NumPy
        # ...
        return np.random.rand(10)  # Ejemplo de datos aleatorios

    def run(self):
        """
        Ejecuta el modelo de entrenamiento y comercio.
        """
        # Iniciar el entrenamiento en un hilo separado
        training_thread = threading.Thread(target=self.train)
        training_thread.start()

        # Realizar operaciones de comercio mientras se entrena el modelo
        while not self.training_stopped:
            # Obtener la lista de símbolos de futuros perpetuos
            futures_list = self.binance_adapter.get_perpetual_futures_list()  # Llama al método get_perpetual_futures_list de BinanceAdp

            # Iterar sobre los símbolos y realizar operaciones de comercio
            for symbol in futures_list:
                # Obtener el apalancamiento óptimo
                leverage = self.get_optimal_leverage(symbol)

                # Realizar una operación de comercio
                self.trade(symbol, "15s", 10800, leverage)  # 3 horas en segundos (10800), velas de 15 segundos ("15s") y 1000 velas

            # Esperar un poco antes de la siguiente iteración
            time.sleep(1)

        # Calcular y mostrar el porcentaje de ganancia total
        if self.total_trades > 0:
            total_profit_percentage = (self.total_profit / self.total_trades) * 100
            print(f"Porcentaje de ganancia total: {total_profit_percentage:.2f}%")
        else:
            print("No se realizaron operaciones de comercio.")

    def parse_tfrecord(self, example_proto):
        """
        Parses a TFRecord example.

        Args:
            example_proto (tf.train.Example): The TFRecord example.

        Returns:
            tuple: A tuple containing the features and target.
        """
        features = {
            "timestamp": tf.io.FixedLenFeature([], tf.int64),
            "open": tf.io.FixedLenFeature([], tf.float32),
            "high": tf.io.FixedLenFeature([], tf.float32),
            "low": tf.io.FixedLenFeature([], tf.float32),
            "close": tf.io.FixedLenFeature([], tf.float32),
            "volume": tf.io.FixedLenFeature([], tf.int64),
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        return parsed_features["close"], parsed_features["close"]  # Assuming you want to predict the next close price

