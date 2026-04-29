import json
import logging
import os
import re
from datetime import datetime
from functools import wraps

from flask import Flask, jsonify, request, abort, send_from_directory, render_template, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from backend.models.database import DatabaseManager, BacktestResult, PerformanceMetric
from backend.strategies.pair_strategy import PairTradingStrategy
from backend.services.backtest_service import BacktestEngine
from backend.execution.execution_service import ExecutionService
from config.settings import Config

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder="../frontend/static",
        template_folder="../frontend/templates",
    )
    app.config["SECRET_KEY"] = Config.SECRET_KEY
    app.config["JSON_SORT_KEYS"] = False

    CORS(app, origins=["http://localhost:5000", "http://127.0.0.1:5000"])

    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["200 per minute"],
        storage_uri="memory://",
    )

    db = DatabaseManager.get_instance()

    # Shared state (single-process dev server)
    _state = {
        "strategy": None,
        "running": False,
        "execution": ExecutionService(),
    }

    # ─── Security helpers ────────────────────────────────────────────────────

    def sanitize_symbol(sym: str) -> str:
        """Strip anything that isn't alphanumeric, =, -, or /"""
        return re.sub(r"[^A-Za-z0-9=\-/]", "", sym)[:20]

    def safe_json(data: dict):
        import math, json as _json
        def clean_nans(obj):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return 0.0
            if isinstance(obj, dict):
                return {k: clean_nans(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [clean_nans(i) for i in obj]
            return obj
        resp = jsonify({"status": "ok", "data": clean_nans(data)})
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Content-Security-Policy"] = "default-src 'self'"
        return resp

    def error_response(msg: str, code: int = 400):
        return jsonify({"status": "error", "message": msg}), code

    # ─── Routes ─────────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/health")
    def health():
        return safe_json({"healthy": True, "timestamp": datetime.utcnow().isoformat()})

    @app.route("/api/start-bot", methods=["POST"])
    @limiter.limit("5 per minute")
    def start_bot():
        body = request.get_json(silent=True) or {}
        sym_x = sanitize_symbol(body.get("symbol_x", Config.PRIMARY_PAIR[0]))
        sym_y = sanitize_symbol(body.get("symbol_y", Config.PRIMARY_PAIR[1]))
        period = body.get("period", "5y")

        if period not in ("1y", "2y", "3y", "5y"):
            return error_response("Invalid period. Use: 1y, 2y, 3y, 5y")

        try:
            strategy = PairTradingStrategy(symbol_x=sym_x, symbol_y=sym_y)
            init_result = strategy.initialize(period=period)
            _state["strategy"] = strategy
            _state["running"] = True
            logger.info(f"Bot started: {sym_x}/{sym_y}")
            return safe_json(init_result)
        except Exception as exc:
            logger.error(f"start-bot error: {exc}")
            return error_response("Failed to start bot", 500)

    @app.route("/api/stop-bot", methods=["POST"])
    def stop_bot():
        _state["running"] = False
        _state["strategy"] = None
        logger.info("Bot stopped")
        return safe_json({"running": False})

    @app.route("/api/signal", methods=["GET"])
    def get_signal():
        if not _state["running"] or not _state["strategy"]:
            return error_response("Bot not running", 400)
        try:
            signal_data = _state["strategy"].get_latest_signal()
            _state["execution"].save_signal(signal_data)
            return safe_json(signal_data)
        except Exception as exc:
            logger.error(f"Signal error: {exc}")
            return error_response("Signal generation failed", 500)

    @app.route("/api/run-backtest", methods=["POST"])
    @limiter.limit("3 per minute")
    def run_backtest():
        body = request.get_json(silent=True) or {}
        sym_x = sanitize_symbol(body.get("symbol_x", Config.PRIMARY_PAIR[0]))
        sym_y = sanitize_symbol(body.get("symbol_y", Config.PRIMARY_PAIR[1]))
        period = body.get("period", "5y")
        walk_forward = bool(body.get("walk_forward", False))

        if period not in ("1y", "2y", "3y", "5y"):
            return error_response("Invalid period")

        try:
            engine = BacktestEngine()

            if walk_forward:
                wf_results = engine.walk_forward(sym_x, sym_y, period=period)
                return safe_json({"walk_forward_results": wf_results})

            result = engine.run(sym_x, sym_y, period=period)
            metrics = result["metrics"]

            # Persist to DB
            session = db.get_session()
            try:
                bt = BacktestResult(
                    pair_symbol=f"{sym_x}/{sym_y}",
                    start_date=result["start_date"],
                    end_date=result["end_date"],
                    initial_capital=engine.initial_capital,
                    final_capital=metrics.get("final_capital", engine.initial_capital),
                    total_return_pct=metrics.get("total_return_pct"),
                    sharpe_ratio=metrics.get("sharpe_ratio"),
                    max_drawdown_pct=metrics.get("max_drawdown_pct"),
                    win_rate=metrics.get("win_rate"),
                    profit_factor=metrics.get("profit_factor"),
                    total_trades=metrics.get("total_trades"),
                    transaction_cost_pct=engine.tc,
                    slippage_pct=engine.slippage,
                    equity_curve=json.dumps(result["equity_curve"][-500:]),
                    trade_log=json.dumps(result["trade_log"][-100:]),
                )
                session.add(bt)
                session.commit()
                result["backtest_id"] = bt.id
            finally:
                session.close()

            return safe_json(result)
        except Exception as exc:
            logger.error(f"Backtest error: {exc}")
            return error_response("Backtest failed", 500)

    @app.route("/api/get-metrics", methods=["GET"])
    def get_metrics():
        session = db.get_session()
        try:
            backtests = (
                session.query(BacktestResult)
                .order_by(BacktestResult.created_at.desc())
                .limit(10)
                .all()
            )
            return safe_json({"backtests": [b.to_dict() for b in backtests]})
        finally:
            session.close()

    @app.route("/api/get-trades", methods=["GET"])
    def get_trades():
        limit = min(int(request.args.get("limit", 50)), 200)
        trades = _state["execution"].get_trade_history(limit=limit)
        open_trades = _state["execution"].get_open_trades()
        return safe_json({"open": open_trades, "history": trades})

    @app.route("/api/ghs-rates", methods=["GET"])
    def ghs_rates():
        from backend.data.data_handler import DataHandler
        try:
            rates = DataHandler().get_ghs_rates()
            return safe_json(rates)
        except Exception as exc:
            logger.error(f"GHS rates error: {exc}")
            return error_response("Rates unavailable", 500)

    @app.route("/api/pair-info", methods=["GET"])
    def pair_info():
        if not _state["strategy"]:
            return safe_json({"initialized": False})
        metrics = _state["strategy"].get_pair_metrics()
        signal = _state["strategy"].get_latest_signal() if _state["running"] else {}
        return safe_json({"pair_metrics": metrics, "latest_signal": signal})

    @app.errorhandler(404)
    def not_found(e):
        return error_response("Not found", 404)

    @app.errorhandler(429)
    def rate_limited(e):
        return error_response("Rate limit exceeded", 429)

    @app.errorhandler(500)
    def server_error(e):
        logger.error(f"Unhandled 500: {e}")
        return error_response("Internal server error", 500)

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(debug=False, host="0.0.0.0", port=5000)
