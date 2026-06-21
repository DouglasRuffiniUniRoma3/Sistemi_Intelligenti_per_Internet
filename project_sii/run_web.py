# Sviluppo completo di run_web.py
from src.web.app import app

if __name__ == "__main__":
    print("[Interface Layer] Avvio della Dashboard Web Intelligente...")
    app.run(debug=True, port=5000)