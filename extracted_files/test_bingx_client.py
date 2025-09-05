```python
import os
from bingx_client import BingXClient

def main():
    # Charger les clés API depuis variables d'environnement
    api_key = os.getenv("BINGX_API_KEY")
    api_secret = os.getenv("BINGX_API_SECRET")
    source_key = os.getenv("BINGX_SOURCE_KEY") # facultatif

    if not api_key or not api_secret:
        print("⚠️ Merci de définir BINGX_API_KEY et BINGX_API_SECRET dans ton environnement")
        return

    # Initialiser client (testnet=True si dispo)
    client = BingXClient(api_key, api_secret, source_key=source_key, testnet=True)

    # 1. Vérifier le solde
    print("\n=== Vérification du solde ===")
    try:
        balance = client.get_balance()
        print("Balance:", balance)
    except Exception as e:
        print("Erreur lors de get_balance:", e)

    # 2. Tester un ordre (⚠️ uniquement en testnet, pas en réel !)
    print("\n=== Placement d’un ordre testnet ===")
    try:
        order = client.place_order(
            symbol="BTC-USDT",
            side="BUY",
            order_type="MARKET",
            quantity=0.001 # petite taille pour test
        )
        print("Order:", order)
    except Exception as e:
        print("Erreur lors de place_order:", e)

if __name__ == "__main__":
    main()
```

---

### ⚙️ Utilisation

1. Crée un fichier `.env` avec tes clés :

```
BINGX_API_KEY=ta_cle_api
BINGX_API_SECRET=ton_secret
BINGX_SOURCE_KEY=optionnel
```

2. Charge-les dans ton terminal (Linux/macOS) :

```bash
export BINGX_API_KEY="ta_cle_api"
export BINGX_API_SECRET="ton_secret"
```

3. Lance le test :

```bash
python test_bingx_client.py
```
