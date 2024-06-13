# src/api/tread/BinanceAdp.py

import os
import ccxt
import time
import json

from .BinanceServ import BinanceServ  # Importa la clase BinanceServ
import pandas as pd
import tensorflow as tf

class BinanceAdp:
    """
    Adaptador para BinanceServ, simplifica el uso de la API de Binance.
    """

    def __init__(self, api_key, secret_key):
            """
            Inicializa el adaptador con las claves de la API de Binance.

            Args:
                api_key (str): Clave de la API de Binance.
                secret_key (str): Clave secreta de la API de Binance.
            """
            self.api_key = api_key
            self.secret_key = secret_key
            self.binance = ccxt.binance({
                'apiKey': api_key,
                'ecret': secret_key,
                'enableRateLimit': True,  # 是否启用Rate Limit
            })

    def get_historical_futures_data(self,timeframe, minutes_ago):
        """
        Obtiene datos históricos de OHLCV para un contrato de futuros perpetuo en Binance y los convierte a TFRecords.

        Args:
            symbol (str): Símbolo del contrato de futuros perpetuo (por ejemplo, "BTCUSDT").
            timeframe (str): Intervalo de tiempo para los datos históricos (por ejemplo, "1m", "5m", "1h").
            minutes_ago (int): Número de minutos para retroceder en el tiempo para los datos históricos.

        Returns:
            str: Nombre del archivo TFRecords creado.
                Si se produce un error, devuelve None.
        """

        # Obtener datos históricos de BinanceServ
        historical_data_json = self.binance_service.get_historical_futures_data(symbol, 3, 30*10000)

        if historical_data_json is None:
            return None

        # Convertir a TFRecords
        try:
            # Convertir el JSON a un diccionario
            data_dict = json.loads(historical_data_json)

            # Convertir OHLCV data to Pandas DataFrame
            df = pd.DataFrame(data_dict["data"])
            df.columns = ["timestamp", "open", "high", "low", "close", "volume"]

            # Create TFRecords filename
            filename = f"ohlcv_data_{symbol}_{timeframe}.tfrecords"

            # Create TFRecords writer
            writer = tf.io.TFRecordWriter(filename)

            # Process and write data to TFRecords
            for index, row in df.iterrows():
                # Create example features and example object
                features = {
                    "timestamp": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(row["timestamp"])])),
                    "open": tf.train.Feature(float_list=tf.train.FloatList(value=[row["open"]])),
                    "high": tf.train.Feature(float_list=tf.train.FloatList(value=[row["high"]])),
                    "low": tf.train.Feature(float_list=tf.train.FloatList(value=[row["low"]])),
                    "close": tf.train.Feature(float_list=tf.train.FloatList(value=[row["close"]])),
                    "volume": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(row["volume"])])),
                }

                example = tf.train.Example(features=tf.train.Features(feature=features))

                # Write example to TFRecords file
                writer.write(example.SerializeToString())

            # Close TFRecords writer
            writer.close()

            # Return success message
            return filename

        except Exception as e:
            print(f"Error al convertir a TFRecords: {e}")
            return None

    def get_perpetual_futures_list(self):
        """
        Obtiene una lista de todos los símbolos de futuros perpetuos en Binance.

        Returns:
            list: Una lista de símbolos de futuros perpetuos (por ejemplo, ["BTCUSDT", "ETHUSDT", "BNBUSDT"]).
        """

        return self.binance_service.get_perpetual_futures_list()  # Llama al método de BinanceServ

    # Agrega otras funciones para interactuar con la API de Binance según sea necesario
    # ...
