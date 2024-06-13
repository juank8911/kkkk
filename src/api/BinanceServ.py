# src/api/tread/binanceServ.py

import os
import ccxt
import time
import json
class BinanceServ:
    """
    Binance API adapter
        Crea la conexión con la API de Binance con la key Pass y secret key 
        Devuelve un token que se debe almacenar en memoria
    """

    def __init__(self, api_key, secret_key):
        """
        Inicializa la conexión con la API de Binance.

        Args:
            api_key (str): La clave API de Binance.
            secret_key (str): La clave secreta de Binance.
        """
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret_key,
        })

    def get_historical_futures_data(self, symbol, timeframe, minutes_ago):
        """
        Obtiene datos históricos de OHLCV para un contrato de futuros perpetuo en Binance.

        Args:
            symbol (str): Símbolo del contrato de futuros perpetuo (por ejemplo, "BTCUSDT").
            timeframe (str): Intervalo de tiempo para los datos históricos (por ejemplo, "1m", "5m", "1h").
            minutes_ago (int): Número de minutos para retroceder en el tiempo para los datos históricos.

        Returns:
            dict: Un diccionario que contiene los datos históricos de OHLCV en formato JSON.
                Si se produce un error, devuelve None.
        """

        try:
            # Filtrar mercados para asegurarse de que el símbolo solicitado sea un futuro perpetuo
            perpetual_futures = [
                market for market in self.exchange.fetch_markets() if market['contractType'] == 'PERPETUAL'
            ]
            if symbol not in [market['symbol'] for market in perpetual_futures]:
                print(f"Error: El símbolo '{symbol}' no es un contrato de futuros perpetuo en Binance.")
                return None

            # Calcular la marca de tiempo para 'minutes_ago' antes de la hora actual
            now = int(time.time())
            since = now - (minutes_ago * 60)

            # Obtener datos históricos de OHLCV
            ohlcv_data = self.exchange.fetch_ohlcv(symbol, timeframe, since=since)

            # Crear un diccionario para almacenar los datos
            data_dict = {
                "symbol": symbol,
                "timeframe": timeframe,
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
