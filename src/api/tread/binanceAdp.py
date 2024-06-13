# src/api/tread/binanceAdp.py

import os
import ccxt
import time

class BinanceAdp:
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

    def get_tokenHiistorico(self):
        """
        Obtiene un token de autenticación de la API de Binance.

        Returns:
            str: El token de autenticación.
        """
        try:
            # Realiza la solicitud de autenticación a la API de Binance
            
            # (El método específico para obtener el token puede variar según la API de Binance)
            # ...

            # Ejemplo de obtención de un token (reemplaza con la lógica real)
            token = self.exchange.privateGetAccount()  # Reemplaza con el método correcto
            return token['token']  # Reemplaza con el campo correcto del token
        

        except Exception as e:
            print(f"Error al obtener el token: {e}")
            return None

    # Agrega otras funciones para interactuar con la API de Binance según sea necesario
    # ...
