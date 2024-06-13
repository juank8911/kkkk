# src/api/tread/binanceServ.py

import os
import ccxt
import time
import json
import dotenv

dotenv.load_dotenv()  # Load environment variables from .env

class BinanceServ:
    """
    Binance API adapter
        Crea la conexión con la API de Binance con la key Pass y secret key 
        Devuelve un token que se debe almacenar en memoria
    """

    def __init__(self):
        """
        Inicializa la conexión con la API de Binance.
        """
        api_key = os.getenv('api_key')
        secret_key = os.getenv('secret_key')
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret_key,
        })

    def get_historical_futures_data(self):
        """
        Obtiene datos históricos de OHLCV para todos los contratos de futuros perpetuos en Binance.

        Returns:
            dict: Un diccionario que contiene los datos históricos de OHLCV en formato JSON.
                Si se produce un error, devuelve None.
        """

        try:
            # Filtrar mercados para asegurarse de que el símbolo solicitado sea un futuro perpetuo
            perpetual_futures = [
                market for market in self.exchange.fetch_markets() if market['contractType'] == 'PERPETUAL'
            ]

            # Crear un diccionario para almacenar los datos
            data_dict = {}

            # Iterar sobre los símbolos de futuros perpetuos
            for market in perpetual_futures:
                symbol = market['symbol']

                # Calcular la marca de tiempo para 'minutes_ago' antes de la hora actual
                now = int(time.time() * 1000)  # Marca de tiempo en milisegundos
                since = now - (3 * 60 * 60 * 1000)  # 3 horas en milisegundos

                # Obtener datos históricos de OHLCV
                ohlcv_data = self.exchange.fetch_ohlcv(symbol,90,[1800, 100])  # Velas de 30 segundos, límite de 1000 velas

                # Agregar los datos al diccionario
                data_dict[symbol] = {
                    "timeframe": "30s",
                    "data": ohlcv_data
                }

            # Convertir el diccionario a formato JSON
            historical_data_json = json.dumps(data_dict)

            # Devolver el JSON con los datos históricos
            return historical_data_json

        except Exception as e:
            print(f"Error al obtener datos históricos: {e}")
            return None

    def get_perpetual_futures_list(self):
        """
        Obtiene una lista de todos los símbolos de futuros perpetuos en Binance.

        Returns:
            list: Una lista de símbolos de futuros perpetuos (por ejemplo, ["BTCUSDT", "ETHUSDT", "BNBUSDT"]).
        """

        try:
            # Obtener todos los mercados de Binance
            markets = self.exchange.fetch_markets()

            # Filtrar y devolver símbolos de futuros perpetuos
            perpetual_futures_symbols = [ market['symbol'] for market in markets if market['contractType'] == 'PERPETUAL'
            ]
            return perpetual_futures_symbols

        except Exception as e:
            print(f"Error al obtener la lista de futuros perpetuos: {e}")
            return None

    # Agrega otras funciones para interactuar con la API de Binance según sea necesario
    # ...

