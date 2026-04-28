// charts.js — Chart.js wrappers
const Charts = (() => {
  const GRID_COLOR = "rgba(48,54,61,0.6)";
  const TEXT_COLOR = "#8b949e";
  const GREEN = "#3fb950";
  const RED = "#f85149";
  const BLUE = "#58a6ff";
  const YELLOW = "#d29922";

  Chart.defaults.color = TEXT_COLOR;
  Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";
  Chart.defaults.font.size = 11;

  let equityChart = null;
  let btEquityChart = null;
  let zscoreChart = null;

  function buildEquityChart(canvasId, labels, data, label = "Equity (GHS)") {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    if (Chart.getChart(ctx)) Chart.getChart(ctx).destroy();

    const isProfit = data.length > 1 && data[data.length - 1] >= data[0];
    const lineColor = isProfit ? GREEN : RED;

    return new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [{
          label,
          data,
          borderColor: lineColor,
          backgroundColor: lineColor + "18",
          fill: true,
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.3,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        animation: { duration: 400 },
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#161b22",
            borderColor: "#30363d",
            borderWidth: 1,
            callbacks: {
              label: (ctx) => ` ${Number(ctx.raw).toLocaleString("en-GH", {
                style: "currency", currency: "GHS", maximumFractionDigits: 0
              })}`,
            },
          },
        },
        scales: {
          x: {
            ticks: { maxTicksLimit: 8, maxRotation: 0 },
            grid: { color: GRID_COLOR },
          },
          y: {
            ticks: {
              callback: (v) => "GHS " + Number(v).toLocaleString(),
            },
            grid: { color: GRID_COLOR },
          },
        },
      },
    });
  }

  function buildZScoreGauge(canvasId, zscore) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    if (Chart.getChart(ctx)) Chart.getChart(ctx).destroy();

    const z = isNaN(zscore) ? 0 : Math.max(-4, Math.min(4, zscore));
    // Normalize 0-100 for doughnut
    const pct = ((z + 4) / 8) * 100;

    let color = BLUE;
    if (zscore > 2) color = RED;
    else if (zscore < -2) color = GREEN;
    else if (Math.abs(zscore) > 1) color = YELLOW;

    return new Chart(ctx, {
      type: "doughnut",
      data: {
        datasets: [{
          data: [pct, 100 - pct],
          backgroundColor: [color, "#21262d"],
          borderWidth: 0,
          circumference: 270,
          rotation: 225,
        }],
      },
      options: {
        responsive: false,
        cutout: "70%",
        animation: { duration: 600 },
        plugins: {
          legend: { display: false },
          tooltip: { enabled: false },
        },
      },
      plugins: [{
        id: "center-text",
        afterDraw(chart) {
          const { ctx: c, width, height } = chart;
          c.save();
          c.font = "bold 28px 'Segoe UI', system-ui";
          c.fillStyle = color;
          c.textAlign = "center";
          c.textBaseline = "middle";
          c.fillText(isNaN(zscore) ? "—" : zscore.toFixed(2), width / 2, height / 2 + 10);
          c.font = "11px 'Segoe UI'";
          c.fillStyle = "#8b949e";
          c.fillText("Z-Score", width / 2, height / 2 + 34);
          c.restore();
        },
      }],
    });
  }

  return { buildEquityChart, buildZScoreGauge };
})();
