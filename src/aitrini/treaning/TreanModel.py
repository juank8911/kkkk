# src/aitrini/treaning/TreanModel.py
import tensorflow as tf
import pandas as pd
from src.aitrini.base.Trinity import create_model  # Importar la función create_model
from ...api.BinanceAdp import BinanceAdp  # Importar la clase BinanceAdp
import time
import ccxt

class TreanModel:
    def __init__(self, api_key, secret_key, tfrecords_path, input_shape, target_shape, learning_rate=0.001):
        """
        Inicializa el modelo de entrenamiento.

        Args:
            api_key (str): La clave API de Binance.
            secret_key (str): La clave secreta de Binance.
            tfrecords_path (str): La ruta al archivo TFRecords.
            input_shape (tuple): La forma de los datos de entrada.
            target_shape (tuple): La forma de los datos de salida.
            learning_rate (float): La tasa de aprendizaje del optimizador.
        """
        self.api_key = api_key  #llamar apiKey dese .env
        self.secret_key = secret_key
        self.tfrecords_path = tfrecords_path
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

        # Cargar los datos de entrenamiento desde TFRecords
        self.train_dataset = self.load_tfrecords(tfrecords_path)

    def load_tfrecords(self, tfrecords_path):
        """
        Carga los datos de entrenamiento desde un archivo TFRecords.

        Args:
            tfrecords_path (str): La ruta al archivo TFRecords.

        Returns:
            tf.data.Dataset: Un conjunto de datos de TensorFlow.
        """
        return tf.data.TFRecordDataset(tfrecords_path).map(self.parse_tfrecord)

    def parse_tfrecord(self, example_proto):
        """
        Analiza un ejemplo de TFRecords.

        Args:
            example_proto (tf.train.Example): Un ejemplo de TFRecords.

        Returns:
            tuple: Un par de tensores que representan los datos de entrada y salida.
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
        return parsed_features["timestamp"], parsed_features["close"]

    def train(self, epochs=10, batch_size=32):
        """
        Entrena el modelo.

        Args:
            epochs (int): El número de épocas de entrenamiento.
            batch_size (int): El tamaño del lote de entrenamiento.
        """
        self.model.fit(self.train_dataset, epochs=epochs, batch_size=batch_size)

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
        historical_data = self.binance_adapter.get_historical_futures_data(symbol, timeframe, minutes_ago)

        # Convertir los datos a un DataFrame de Pandas
        df = pd.DataFrame(historical_data["data"])
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]

        # Realizar una predicción con el modelo
        prediction = self.predict(df["close"].values.reshape(1, -1))

        # Tomar una decisión de compra o venta
        if prediction > 0.5:
            # Comprar
            self.buy(symbol, leverage)
        else:
            # Vender
            self.sell(symbol, leverage)

    def buy(self, symbol, leverage):
        """
        Realiza una operación de compra.

        Args:
            symbol (str): El símbolo del contrato de futuros.
            leverage (int): El apalancamiento a utilizar.
        """
        # Obtener el precio actual
        current_price = self.binance_adapter.exchange.fetch_ticker(symbol)["last"]

        # Calcular el precio objetivo
        target_price = current_price * (1 + 0.0015)  # Ganancia mínima del 0.15%

        # Colocar una orden de compra
        self.binance_adapter.exchange.create_market_buy_order(symbol, current_price, leverage)

        # Esperar a que el precio alcance el precio objetivo
        while self.binance_adapter.exchange.fetch_ticker(symbol)["last"] < target_price:
            time.sleep(1)

        # Cerrar la posición
        self.close_position(symbol)

    def sell(self, symbol, leverage):
        """
        Realiza una operación de venta.

        Args:
            symbol (str): El símbolo del contrato de futuros.
            leverage (int): El apalancamiento a utilizar.
        """
        # Obtener el precio actual
        current_price = self.binance_adapter.exchange.fetch_ticker(symbol)["last"]

        # Calcular el precio objetivo
        target_price = current_price * (1 - 0.0015)  # Ganancia mínima del 0.15%

        # Colocar una orden de venta
        self.binance_adapter.exchange.create_market_sell_order(symbol, current_price, leverage)

        # Esperar a que el precio alcance el precio objetivo
        while self.binance_adapter.exchange.fetch_ticker(symbol)["last"] > target_price:
            time.sleep(1)

        # Cerrar la posición
        self.close_position(symbol)

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
        Calcula el apalancamiento óptimo para un símbolo.

        Args:
            symbol (str): El símbolo del contrato de futuros.

        Returns:
            int: El apalancamiento óptimo.
        """
        # Implementar la lógica para calcular el apalancamiento óptimo
        # ...
        return 10  # Ejemplo de apalancamiento

    def run(self):
        """
        Ejecuta el modelo de entrenamiento y comercio.
        """
        # Entrenar el modelo
        self.train()

        # Obtener la lista de símbolos de futuros perpetuos
        futures_list = self.binance_adapter.get_perpetual_futures_list()

        # Iterar sobre los símbolos y realizar operaciones de comercio
        for symbol in futures_list:
            # Obtener el apalancamiento óptimo
            leverage = self.get_optimal_leverage(symbol)

            # Realizar una operación de comercio
            self.trade(symbol, "1m", 60, leverage)

