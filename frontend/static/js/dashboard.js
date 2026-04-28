// dashboard.js — main UI controller
(async () => {
  // ── State ──────────────────────────────────────────────
  let botRunning = false;
  let signalInterval = null;
  let currentCapital = 50000;

  // ── DOM refs ───────────────────────────────────────────
  const btnStart = document.getElementById("btn-start");
  const btnStop = document.getElementById("btn-stop");
  const btnRefreshSignal = document.getElementById("btn-refresh-signal");
  const btnBacktest = document.getElementById("btn-backtest");
  const btnRefreshTrades = document.getElementById("btn-refresh-trades");
  const botStatusBadge = document.getElementById("bot-status-badge");
  const regimeBadge = document.getElementById("regime-badge");
  const ghsRateDisplay = document.getElementById("ghs-rate-display");
  const capitalDisplay = document.getElementById("capital-display");

  // ── Toast ──────────────────────────────────────────────
  function toast(msg, type = "info") {
    const id = "t" + Date.now();
    const colors = { success: "#3fb950", error: "#f85149", info: "#58a6ff", warning: "#d29922" };
    const html = `
      <div id="${id}" class="toast align-items-center show mb-2" role="alert">
        <div class="d-flex">
          <div class="toast-body">
            <span style="color:${colors[type] || colors.info}">●</span> ${escHtml(msg)}
          </div>
          <button type="button" class="btn-close btn-close-white me-2 m-auto"
            onclick="document.getElementById('${id}').remove()"></button>
        </div>
      </div>`;
    document.getElementById("toast-container").insertAdjacentHTML("beforeend", html);
    setTimeout(() => document.getElementById(id)?.remove(), 5000);
  }

  function escHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  // ── GHS Rates ──────────────────────────────────────────
  async function loadGhsRates() {
    try {
      const rates = await API.getGhsRates();
      ghsRateDisplay.textContent =
        `1 USD = ${Number(rates.USD_GHS).toFixed(2)} GHS | 1 EUR = ${Number(rates.EUR_GHS).toFixed(2)} GHS`;
    } catch {
      ghsRateDisplay.textContent = "Rates unavailable";
    }
  }

  // ── Signal display ─────────────────────────────────────
  function renderSignal(sig) {
    const mSignal = document.getElementById("m-signal");
    const mZscore = document.getElementById("m-zscore");
    const mHedge = document.getElementById("m-hedge");
    const mRegime = document.getElementById("m-regime");

    const signalVal = sig.signal || "—";
    const colorMap = { LONG: "signal-long", SHORT: "signal-short", EXIT: "signal-exit", HOLD: "signal-hold", NONE: "signal-none" };
    mSignal.textContent = signalVal;
    mSignal.className = "metric-value " + (colorMap[signalVal] || "signal-none");

    const z = sig.zscore;
    mZscore.textContent = z != null ? Number(z).toFixed(3) : "—";
    mZscore.className = "metric-value " + (z > 2 ? "signal-short" : z < -2 ? "signal-long" : z != null ? "signal-hold" : "");

    mHedge.textContent = sig.hedge_ratio != null ? Number(sig.hedge_ratio).toFixed(4) : "—";

    const regime = sig.regime || "—";
    mRegime.textContent = regime;
    mRegime.className = "metric-value " + (regime === "MEAN_REVERTING" ? "signal-long" : regime === "TRENDING" ? "signal-short" : "");

    // Navbar regime badge
    regimeBadge.textContent = regime.replace("_", " ");
    regimeBadge.className = "badge " + (regime === "MEAN_REVERTING" ? "regime-mean" : regime === "TRENDING" ? "regime-trend" : "bg-secondary");

    // Z-score gauge
    Charts.buildZScoreGauge("zscore-chart", z != null ? Number(z) : NaN);
  }

  // ── Pair metrics ───────────────────────────────────────
  function renderPairMetrics(metrics) {
    if (!metrics) return;
    const row = document.getElementById("pair-metrics-row");
    const items = [
      ["Cointegrated", metrics.is_cointegrated ? "✓ Yes" : "✗ No", metrics.is_cointegrated ? "text-success" : "text-danger"],
      ["Coint p-value", metrics.coint_pvalue != null ? Number(metrics.coint_pvalue).toFixed(4) : "—", ""],
      ["Stationary", metrics.is_stationary ? "✓ Yes" : "✗ No", metrics.is_stationary ? "text-success" : "text-danger"],
      ["Half-life (days)", metrics.halflife_days != null ? Number(metrics.halflife_days).toFixed(1) : "—", ""],
      ["Hurst exponent", metrics.hurst_exponent != null ? Number(metrics.hurst_exponent).toFixed(3) : "—", ""],
      ["Valid pair", metrics.is_valid ? "✓ Valid" : "✗ Invalid", metrics.is_valid ? "text-success" : "text-danger"],
    ];
    row.innerHTML = items.map(([label, val, cls]) => `
      <div class="col-6 col-md-2">
        <div class="small text-muted">${escHtml(label)}</div>
        <div class="fw-semibold ${cls}">${escHtml(String(val))}</div>
      </div>`).join("");
  }

  // ── Trades table ───────────────────────────────────────
  function renderTrades(tradeData) {
    const tbody = document.getElementById("trades-tbody");
    const all = [...(tradeData.open || []), ...(tradeData.history || [])];
    if (!all.length) {
      tbody.innerHTML = `<tr><td colspan="10" class="text-center text-muted py-3">No trades yet</td></tr>`;
      return;
    }
    tbody.innerHTML = all.map(t => {
      const pnlUsd = t.pnl_usd != null ? Number(t.pnl_usd) : null;
      const pnlGhs = t.pnl_ghs != null ? Number(t.pnl_ghs) : null;
      const pnlClass = pnlGhs > 0 ? "pnl-positive" : pnlGhs < 0 ? "pnl-negative" : "";
      return `<tr>
        <td>${t.id}</td>
        <td><span class="text-info">${escHtml(t.pair_symbol || "")}</span></td>
        <td><span class="${t.direction === "LONG" ? "text-success" : "text-danger"}">${escHtml(t.direction || "")}</span></td>
        <td>${t.entry_price != null ? Number(t.entry_price).toFixed(5) : "—"}</td>
        <td>${t.exit_price != null ? Number(t.exit_price).toFixed(5) : "—"}</td>
        <td class="${pnlClass}">${pnlUsd != null ? pnlUsd.toFixed(4) : "—"}</td>
        <td class="${pnlClass}">${pnlGhs != null ? pnlGhs.toFixed(2) : "—"}</td>
        <td>${t.zscore_entry != null ? Number(t.zscore_entry).toFixed(3) : "—"}</td>
        <td><span class="badge ${t.regime === "MEAN_REVERTING" ? "regime-mean" : "regime-trend"}">${escHtml(t.regime || "—")}</span></td>
        <td><span class="badge ${t.status === "OPEN" ? "bg-success" : t.status === "CLOSED" ? "bg-secondary" : "bg-warning"}">${escHtml(t.status || "")}</span></td>
      </tr>`;
    }).join("");
  }

  // ── Backtest results ───────────────────────────────────
  function renderBacktestResults(result) {
    document.getElementById("backtest-loading").classList.add("d-none");
    document.getElementById("backtest-results").classList.remove("d-none");

    const m = result.metrics || {};
    const metrics = [
      ["Total Return", (m.total_return_pct || 0).toFixed(2) + "%", m.total_return_pct >= 0 ? "text-success" : "text-danger"],
      ["Sharpe Ratio", (m.sharpe_ratio || 0).toFixed(3), ""],
      ["Max Drawdown", (m.max_drawdown_pct || 0).toFixed(2) + "%", "text-warning"],
      ["Win Rate", (m.win_rate || 0).toFixed(1) + "%", ""],
      ["Profit Factor", isFinite(m.profit_factor) ? (m.profit_factor || 0).toFixed(3) : "∞", ""],
      ["Total Trades", m.total_trades || 0, ""],
      ["P&L (GHS)", (m.total_pnl_ghs || 0).toLocaleString(), m.total_pnl_ghs >= 0 ? "text-success" : "text-danger"],
      ["Period", `${result.start_date} → ${result.end_date}`, "text-muted"],
    ];

    document.getElementById("bt-metrics-row").innerHTML = metrics.map(([label, val, cls]) => `
      <div class="col-6 col-md-3">
        <div class="card dark-card metric-card">
          <div class="card-body text-center py-2">
            <div class="metric-label">${escHtml(label)}</div>
            <div class="fw-bold fs-5 ${cls}">${escHtml(String(val))}</div>
          </div>
        </div>
      </div>`).join("");

    // Equity chart
    const eq = result.equity_curve || [];
    const labels = eq.map((_, i) => i % Math.max(1, Math.floor(eq.length / 8)) === 0 ? `Bar ${i}` : "");
    Charts.buildEquityChart("bt-equity-chart", labels, eq, "Backtest Equity (GHS)");

    // Coint info
    const coint = m.coint_metrics || {};
    document.getElementById("bt-coint-info").innerHTML = Object.entries(coint)
      .filter(([k]) => !k.includes("score"))
      .map(([k, v]) => `<div><span class="text-muted">${escHtml(k.replace(/_/g, " "))}:</span> <strong>${typeof v === "number" ? v.toFixed(4) : escHtml(String(v))}</strong></div>`)
      .join("");
  }

  function renderWalkForwardResults(data) {
    document.getElementById("backtest-loading").classList.add("d-none");
    document.getElementById("wf-results").classList.remove("d-none");
    const rows = data.walk_forward_results || [];
    document.getElementById("wf-tbody").innerHTML = rows.map(r => `
      <tr>
        <td>${r.window}</td>
        <td class="text-muted small">${r.start_date} → ${r.end_date}</td>
        <td>${r.total_trades || 0}</td>
        <td class="${(r.total_return_pct || 0) >= 0 ? "text-success" : "text-danger"}">${(r.total_return_pct || 0).toFixed(2)}%</td>
        <td>${(r.sharpe_ratio || 0).toFixed(3)}</td>
        <td class="text-warning">${(r.max_drawdown_pct || 0).toFixed(2)}%</td>
        <td>${(r.win_rate || 0).toFixed(1)}%</td>
      </tr>`).join("");
  }

  // ── Event: Start Bot ───────────────────────────────────
  btnStart.addEventListener("click", async () => {
    const symX = document.getElementById("sym-x").value.trim();
    const symY = document.getElementById("sym-y").value.trim();
    const period = document.getElementById("period-select").value;

    if (!symX || !symY) { toast("Enter both symbols", "warning"); return; }

    btnStart.disabled = true;
    btnStart.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Loading…';

    try {
      const result = await API.startBot(symX, symY, period);
      botRunning = true;
      btnStop.disabled = false;
      btnRefreshSignal.disabled = false;
      botStatusBadge.textContent = "RUNNING";
      botStatusBadge.className = "badge bg-success";
      toast(`Bot started: ${symX}/${symY} (${result.data_points} bars)`, "success");

      if (result.pair_metrics) renderPairMetrics(result.pair_metrics);

      // Auto-refresh signal every 60s
      signalInterval = setInterval(refreshSignal, 60000);
      await refreshSignal();

    } catch (err) {
      toast("Failed to start bot: " + err.message, "error");
    } finally {
      btnStart.disabled = false;
      btnStart.innerHTML = '<i class="bi bi-play-fill"></i> Start Bot';
    }
  });

  // ── Event: Stop Bot ────────────────────────────────────
  btnStop.addEventListener("click", async () => {
    try {
      await API.stopBot();
      botRunning = false;
      clearInterval(signalInterval);
      btnStop.disabled = true;
      btnRefreshSignal.disabled = true;
      botStatusBadge.textContent = "STOPPED";
      botStatusBadge.className = "badge bg-secondary";
      toast("Bot stopped", "warning");
    } catch (err) {
      toast("Stop failed: " + err.message, "error");
    }
  });

  // ── Signal refresh ─────────────────────────────────────
  async function refreshSignal() {
    if (!botRunning) return;
    try {
      const sig = await API.getSignal();
      renderSignal(sig);
    } catch (err) {
      toast("Signal error: " + err.message, "error");
    }
  }

  btnRefreshSignal.addEventListener("click", refreshSignal);

  // ── Event: Backtest ────────────────────────────────────
  btnBacktest.addEventListener("click", async () => {
    const symX = document.getElementById("sym-x").value.trim();
    const symY = document.getElementById("sym-y").value.trim();
    const period = document.getElementById("period-select").value;
    const walkFwd = document.getElementById("wf-toggle").checked;

    document.getElementById("backtest-results").classList.add("d-none");
    document.getElementById("wf-results").classList.add("d-none");
    document.getElementById("backtest-loading").classList.remove("d-none");
    btnBacktest.disabled = true;

    try {
      const result = await API.runBacktest(symX, symY, period, walkFwd);
      if (walkFwd) {
        renderWalkForwardResults(result);
      } else {
        renderBacktestResults(result);
      }
      toast("Backtest complete!", "success");
    } catch (err) {
      document.getElementById("backtest-loading").classList.add("d-none");
      toast("Backtest failed: " + err.message, "error");
    } finally {
      btnBacktest.disabled = false;
    }
  });

  // ── Event: Refresh trades ──────────────────────────────
  btnRefreshTrades.addEventListener("click", async () => {
    try {
      const trades = await API.getTrades(50);
      renderTrades(trades);
    } catch (err) {
      toast("Failed to load trades", "error");
    }
  });

  // ── Init ───────────────────────────────────────────────
  await loadGhsRates();
  Charts.buildZScoreGauge("zscore-chart", NaN);
  await btnRefreshTrades.click();

})();
