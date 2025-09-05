import requests
import hmac
import hashlib
import time

class IA2Executor:
    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://api.bingx.com"

    def sign_request(self, params):
        # Signe la requête pour l'API BingX
        query_string = '&'.join([f"{key}={params[key]}" for key in sorted(params)])
        signature = hmac.new(self.secret_key.encode(), query_string.encode(), hashlib.sha256).hexdigest()
        return signature

    def place_order(self, strategy):
        # Place un ordre sur BingX
        params = {
            'symbol': strategy['symbol'],
            'side': strategy['action'].upper(),
            'type': 'MARKET',
            'quantity': self.calculate_quantity(strategy),
            'timestamp': int(time.time() * 1000)
        }
        params['signature'] = self.sign_request(params)
        response = requests.post(f"{self.base_url}/trade/order", params=params)
        return response.json()

    def calculate_quantity(self, strategy):
        # Calcule la quantité en fonction de la taille de la position et du risque
        # Pour l'exemple, fixe à 1
        return 1

    def monitor_positions(self):
        # Récupère les positions ouvertes
        params = {
            'timestamp': int(time.time() * 1000)
        }
        params['signature'] = self.sign_request(params)
        response = requests.get(f"{self.base_url}/trade/positions", params=params)
        return response.json()

    def adjust_orders(self, position, market_data):
        # Ajuste les ordres en fonction du marché et des conseils de l'IA BingX
        # Logique d'ajustement
        pass

    def close_position(self, symbol):
        # Ferme la position pour le symbole
        params = {
            'symbol': symbol,
            'side': 'SELL',  # ou 'BUY' pour une position courte
            'type': 'MARKET',
            'quantity': 'all',
            'timestamp': int(time.time() * 1000)
        }
        params['signature'] = self.sign_request(params)
        response = requests.post(f"{self.base_url}/trade/order", params=params)
        return response.json()