import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.api.app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    print(f"\n{'='*50}")
    print(f"  GhanaQuant Pair Trading Bot")
    print(f"  Running on http://localhost:{port}")
    print(f"  Mode: {'Development' if debug else 'Production'}")
    print(f"{'='*50}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
