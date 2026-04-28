// api.js — all backend communication
const API = (() => {
  const BASE = "";

  async function _fetch(url, opts = {}) {
    try {
      const resp = await fetch(BASE + url, {
        headers: { "Content-Type": "application/json" },
        ...opts,
      });
      const json = await resp.json();
      if (!resp.ok) throw new Error(json.message || `HTTP ${resp.status}`);
      return json.data;
    } catch (err) {
      console.error(`API error [${url}]:`, err);
      throw err;
    }
  }

  return {
    startBot: (symX, symY, period) =>
      _fetch("/api/start-bot", {
        method: "POST",
        body: JSON.stringify({ symbol_x: symX, symbol_y: symY, period }),
      }),

    stopBot: () => _fetch("/api/stop-bot", { method: "POST" }),

    getSignal: () => _fetch("/api/signal"),

    runBacktest: (symX, symY, period, walkForward) =>
      _fetch("/api/run-backtest", {
        method: "POST",
        body: JSON.stringify({
          symbol_x: symX,
          symbol_y: symY,
          period,
          walk_forward: walkForward,
        }),
      }),

    getMetrics: () => _fetch("/api/get-metrics"),

    getTrades: (limit = 50) => _fetch(`/api/get-trades?limit=${limit}`),

    getGhsRates: () => _fetch("/api/ghs-rates"),

    getPairInfo: () => _fetch("/api/pair-info"),

    health: () => _fetch("/api/health"),
  };
})();
