const PREDICTION_FORM_KEY = "smartEnergy_predictionForm";
const PREDICTION_RESULT_KEY = "smartEnergy_predictionResult";
const FORM_FIELDS = ["temperature", "humidity", "voltage", "current", "device_usage"];
const FIELD_LABELS = {
  temperature: "Temperature",
  humidity: "Humidity",
  voltage: "Voltage",
  current: "Current",
  device_usage: "Device Usage",
};

let predictionData = null;
let chartInstances = { line: null, bar: null, pie: null };

const APPLIANCE_COLORS = ["#3d6fd9", "#7c5cff", "#eab308", "#06b6d4", "#8b5cf6"];
const APPLIANCE_HOVER = ["#5080e8", "#9070ff", "#f5c842", "#22d3ee", "#a78bfa"];
const DEVICE_BAR_TOP = ["#8ee4aa", "#d1d5db", "#c4b5fd", "#fde68a", "#67e8f9"];
const DEVICE_BAR_BOTTOM = ["#6ecf8f", "#9ca3af", "#8b5cf6", "#eab308", "#06b6d4"];
const DEVICE_BAR_HOVER = ["#a0eabb", "#e5e7eb", "#a78bfa", "#fbbf24", "#22d3ee"];

const centerTextPlugin = {
  id: "centerText",
  beforeDraw(chart) {
    const meta = chart.getDatasetMeta(0);
    if (!meta?.data?.length) {
      return;
    }

    const dataset = chart.data.datasets[0];
    const labels = chart.data.labels || [];
    const values = dataset.data || [];
    const maxValue = Math.max(...values);
    const maxIndex = values.indexOf(maxValue);
    const topLabel = labels[maxIndex] || "Usage";
    const { ctx, chartArea } = chart;

    if (!chartArea) {
      return;
    }

    const centerX = (chartArea.left + chartArea.right) / 2;
    const centerY = (chartArea.top + chartArea.bottom) / 2;

    ctx.save();
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#9eaccb";
    ctx.font = "500 11px Inter, sans-serif";
    ctx.fillText("Highest share", centerX, centerY - 18);
    ctx.fillStyle = "#eaf1ff";
    ctx.font = "700 26px Inter, sans-serif";
    ctx.fillText(`${maxValue}%`, centerX, centerY + 6);
    ctx.fillStyle = APPLIANCE_COLORS[maxIndex] || "#3d6fd9";
    ctx.font = "600 12px Inter, sans-serif";
    ctx.fillText(topLabel, centerX, centerY + 28);
    ctx.restore();
  },
};

function buildApplianceChart(canvas, labels, values) {
  return new Chart(canvas, {
    type: "doughnut",
    data: {
      labels,
      datasets: [
        {
          data: values,
          backgroundColor: APPLIANCE_COLORS,
          hoverBackgroundColor: APPLIANCE_HOVER,
          borderColor: "#0a0a0a",
          borderWidth: 4,
          borderRadius: 6,
          hoverOffset: 14,
          spacing: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "62%",
      layout: { padding: 8 },
      animation: {
        animateRotate: true,
        animateScale: true,
        duration: 1000,
        easing: "easeOutQuart",
      },
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            padding: 18,
            boxWidth: 10,
            boxHeight: 10,
            usePointStyle: true,
            pointStyle: "circle",
            color: "#eaf1ff",
            font: { size: 12, family: "Inter, sans-serif", weight: "500" },
            generateLabels(chart) {
              const data = chart.data;
              const dataset = data.datasets[0];
              return (data.labels || []).map((label, index) => ({
                text: `${label}   ${dataset.data[index]}%`,
                fillStyle: APPLIANCE_COLORS[index],
                strokeStyle: "transparent",
                fontColor: "#eaf1ff",
                color: "#eaf1ff",
                hidden: false,
                index,
              }));
            },
          },
        },
        tooltip: {
          backgroundColor: "rgba(16, 16, 16, 0.95)",
          borderColor: "rgba(255,255,255,0.12)",
          borderWidth: 1,
          titleColor: "#eaf1ff",
          bodyColor: "#9eaccb",
          padding: 12,
          cornerRadius: 10,
          displayColors: true,
          callbacks: {
            label(context) {
              const total = context.dataset.data.reduce((sum, val) => sum + val, 0);
              const pct = total ? ((context.parsed / total) * 100).toFixed(1) : 0;
              return ` ${context.label}: ${context.parsed}% (${pct}% of total)`;
            },
          },
        },
      },
    },
    plugins: [centerTextPlugin],
  });
}

const barValuePlugin = {
  id: "barValues",
  afterDatasetsDraw(chart) {
    const { ctx } = chart;
    const meta = chart.getDatasetMeta(0);
    if (!meta?.data?.length) {
      return;
    }

    meta.data.forEach((bar, index) => {
      const value = chart.data.datasets[0].data[index];
      ctx.save();
      ctx.fillStyle = "#eaf1ff";
      ctx.font = "600 11px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(`${value}%`, bar.x, bar.y - 6);
      ctx.restore();
    });
  },
};

function buildDeviceBarChart(canvas, labels, values) {
  return new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Device Consumption",
          data: values,
          borderRadius: { topLeft: 14, topRight: 14, bottomLeft: 4, bottomRight: 4 },
          borderSkipped: false,
          maxBarThickness: 52,
          hoverBackgroundColor: DEVICE_BAR_HOVER,
          backgroundColor(context) {
            const chart = context.chart;
            const { ctx, chartArea } = chart;
            const index = context.dataIndex;

            if (!chartArea) {
              return DEVICE_BAR_BOTTOM[index % DEVICE_BAR_BOTTOM.length];
            }

            const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
            gradient.addColorStop(0, DEVICE_BAR_TOP[index % DEVICE_BAR_TOP.length]);
            gradient.addColorStop(1, DEVICE_BAR_BOTTOM[index % DEVICE_BAR_BOTTOM.length]);
            return gradient;
          },
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: { top: 24 } },
      animation: {
        duration: 900,
        easing: "easeOutQuart",
        delay(context) {
          return context.dataIndex * 80;
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(16, 16, 16, 0.95)",
          borderColor: "rgba(255,255,255,0.12)",
          borderWidth: 1,
          titleColor: "#eaf1ff",
          bodyColor: "#9eaccb",
          padding: 12,
          cornerRadius: 10,
          callbacks: {
            label(context) {
              return ` ${context.parsed.y}% of total load`;
            },
          },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          border: { display: false },
          ticks: {
            color: "#eaf1ff",
            font: { size: 11, family: "Inter, sans-serif", weight: "500" },
            maxRotation: 0,
          },
        },
        y: {
          beginAtZero: true,
          border: { display: false },
          grid: {
            color: "rgba(255,255,255,0.06)",
            drawTicks: false,
          },
          ticks: {
            color: "#9eaccb",
            padding: 10,
            font: { size: 10, family: "Inter, sans-serif" },
            callback(value) {
              return `${value}%`;
            },
          },
        },
      },
    },
    plugins: [barValuePlugin],
  });
}

function buildLineChart(canvas, labels, values) {
  return new Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "kWh",
          data: values,
          borderColor: "#6ecf8f",
          backgroundColor: "rgba(110, 207, 143, 0.15)",
          tension: 0.4,
          fill: true,
          pointRadius: 5,
          pointHoverRadius: 7,
          pointBackgroundColor: "#6ecf8f",
          pointBorderColor: "#0a0a0a",
          pointBorderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 900, easing: "easeOutQuart" },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(16, 16, 16, 0.95)",
          borderColor: "rgba(255,255,255,0.12)",
          borderWidth: 1,
          titleColor: "#eaf1ff",
          bodyColor: "#9eaccb",
          padding: 12,
          cornerRadius: 10,
          callbacks: {
            label(context) {
              return ` ${context.parsed.y} kWh`;
            },
          },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          border: { display: false },
          ticks: { color: "#eaf1ff", font: { size: 11, family: "Inter, sans-serif" } },
        },
        y: {
          beginAtZero: false,
          border: { display: false },
          grid: { color: "rgba(255,255,255,0.06)" },
          ticks: { color: "#9eaccb", padding: 8, font: { size: 10, family: "Inter, sans-serif" } },
        },
      },
    },
  });
}

function isReportsPage() {
  return document.getElementById("weekSelector") !== null;
}

let reportsChartInstances = { line: null, bar: null };

function destroyReportsCharts() {
  reportsChartInstances.line?.destroy();
  reportsChartInstances.bar?.destroy();
  reportsChartInstances = { line: null, bar: null };
}

function setActiveWeekUI(weekKey) {
  const reports = window.reportsData;
  const week = reports?.weeks?.[weekKey];

  document.querySelectorAll(".week-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.week === weekKey);
  });
  document.querySelectorAll(".week-row").forEach((row) => {
    row.classList.toggle("active", row.dataset.week === weekKey);
  });
  setText("weekChartTitle", `Daily Trend — ${weekKey}`);

  if (week) {
    setText("weeklyEfficiencyScore", `${week.efficiency_score}%`);
    setText("weeklyEfficiencyLabel", `Weekly Efficiency Score (${weekKey})`);
  }
}

function buildReportsWeekCharts(weekKey) {
  const reports = window.reportsData;
  if (!reports?.weeks?.[weekKey] || !window.Chart) {
    return;
  }

  const week = reports.weeks[weekKey];
  destroyReportsCharts();
  setActiveWeekUI(weekKey);

  Chart.defaults.color = "#dce7ff";
  Chart.defaults.borderColor = "rgba(255,255,255,0.12)";

  const lineCanvas = document.getElementById("lineChart");
  if (lineCanvas) {
    reportsChartInstances.line = buildLineChart(
      lineCanvas,
      week.daily.labels,
      week.daily.values
    );
  }

  const barCanvas = document.getElementById("barChart");
  if (barCanvas) {
    reportsChartInstances.bar = buildDeviceBarChart(
      barCanvas,
      week.device.labels,
      week.device.values
    );
  }
}

function initReportsPage() {
  const defaultWeek = window.reportsData?.default_week || "Week 1";
  buildReportsWeekCharts(defaultWeek);

  document.querySelectorAll(".week-tab").forEach((tab) => {
    tab.addEventListener("click", () => buildReportsWeekCharts(tab.dataset.week));
  });

  document.querySelectorAll(".week-row").forEach((row) => {
    row.addEventListener("click", () => buildReportsWeekCharts(row.dataset.week));
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        buildReportsWeekCharts(row.dataset.week);
      }
    });
  });
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) {
    el.textContent = value;
  }
}

function isDashboardPage() {
  return document.getElementById("analyticsSection") !== null;
}

function clearSavedPredictionState() {
  localStorage.removeItem(PREDICTION_FORM_KEY);
  localStorage.removeItem(PREDICTION_RESULT_KEY);
}

function isPageReload() {
  const entry = performance.getEntriesByType("navigation")[0];
  return entry && entry.type === "reload";
}

function savePredictionForm(form) {
  if (!form) {
    return;
  }
  const values = {};
  FORM_FIELDS.forEach((field) => {
    const input = form.elements[field];
    if (input) {
      values[field] = input.value;
    }
  });
  localStorage.setItem(PREDICTION_FORM_KEY, JSON.stringify(values));
}

function restorePredictionForm(form) {
  if (!form) {
    return;
  }
  const raw = localStorage.getItem(PREDICTION_FORM_KEY);
  if (!raw) {
    return;
  }
  try {
    const values = JSON.parse(raw);
    FORM_FIELDS.forEach((field) => {
      const input = form.elements[field];
      if (input && values[field] !== undefined && values[field] !== "") {
        input.value = values[field];
      }
    });
  } catch {
    localStorage.removeItem(PREDICTION_FORM_KEY);
  }
}

function savePredictionResult(data) {
  localStorage.setItem(PREDICTION_RESULT_KEY, JSON.stringify(data));
}

function showElement(id) {
  const el = document.getElementById(id);
  if (el) {
    el.classList.remove("is-hidden");
  }
}

function hideElement(id) {
  const el = document.getElementById(id);
  if (el) {
    el.classList.add("is-hidden");
  }
}

function showValidationMessage(message) {
  const el = document.getElementById("formValidation");
  if (!el) {
    return;
  }
  el.textContent = message;
  el.hidden = false;
}

function hideValidationMessage() {
  const el = document.getElementById("formValidation");
  if (el) {
    el.textContent = "";
    el.hidden = true;
  }
}

function validateForm(form) {
  const missing = [];
  FORM_FIELDS.forEach((field) => {
    const input = form.elements[field];
    if (!input || String(input.value).trim() === "") {
      missing.push(FIELD_LABELS[field]);
    }
  });

  if (missing.length === FORM_FIELDS.length) {
    return "Please fill in all energy parameters before predicting.";
  }
  if (missing.length > 0) {
    return `Please fill in: ${missing.join(", ")}.`;
  }

  for (const field of FORM_FIELDS) {
    const value = Number(form.elements[field].value);
    if (Number.isNaN(value)) {
      return `${FIELD_LABELS[field]} must be a valid number.`;
    }
  }

  return null;
}

function updateRecommendations(items) {
  const list = document.getElementById("recommendationsList");
  if (!list || !Array.isArray(items)) {
    return;
  }
  list.innerHTML = "";
  items.forEach((tip) => {
    const li = document.createElement("li");
    li.textContent = tip;
    list.appendChild(li);
  });
}

function updateMeter(score) {
  const clamped = Math.max(0, Math.min(100, Number(score)));
  const meter = document.getElementById("meterFill");
  if (meter) {
    meter.style.width = `${clamped}%`;
  }
  setText("efficiencyLabel", `${clamped}%`);
  setText("meterScoreLabel", `${clamped}%`);
}

function destroyCharts() {
  Object.values(chartInstances).forEach((chart) => chart?.destroy());
  chartInstances = { line: null, bar: null, pie: null };
}

function buildChartsFromAnalytics(analytics) {
  if (!analytics || !window.Chart) {
    return;
  }

  destroyCharts();

  Chart.defaults.color = "#dce7ff";
  Chart.defaults.borderColor = "rgba(255,255,255,0.12)";

  const lineCanvas = document.getElementById("lineChart");
  if (lineCanvas) {
    chartInstances.line = new Chart(lineCanvas, {
      type: "line",
      data: {
        labels: analytics.charts.daily.labels,
        datasets: [
          {
            label: "kWh",
            data: analytics.charts.daily.values,
            borderColor: "#b7bdc8",
            backgroundColor: "rgba(183,189,200,0.22)",
            tension: 0.35,
            fill: true,
            pointRadius: 4,
            pointBackgroundColor: "#b7bdc8",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 900, easing: "easeOutQuart" },
        plugins: { legend: { display: false } },
      },
    });
  }

  const barCanvas = document.getElementById("barChart");
  if (barCanvas) {
    chartInstances.bar = buildDeviceBarChart(
      barCanvas,
      analytics.charts.device.labels,
      analytics.charts.device.values
    );
  }

  const pieCanvas = document.getElementById("pieChart");
  if (pieCanvas) {
    chartInstances.pie = buildApplianceChart(
      pieCanvas,
      analytics.charts.appliance.labels,
      analytics.charts.appliance.values
    );
  }
}

function renderAnalytics(data) {
  if (!data || !data.analytics) {
    return;
  }

  predictionData = data;
  const { analytics } = data;
  const cards = analytics.cards;
  const insights = analytics.insights;

  hideElement("emptyState");
  showElement("predictionCard");
  showElement("analyticsSection");

  setText("totalConsumption", `${cards.total_consumption} kWh`);
  setText("predictedConsumption", `${cards.predicted_next_day} kWh`);
  setText("peakUsageHours", cards.peak_usage_hours);
  setText("insightPeakDay", `${insights.peak_day} (${insights.peak_value} kWh)`);
  setText("insightLowDay", `${insights.lowest_day} (${insights.lowest_value} kWh)`);
  setText("insightAvgDaily", `${insights.average_daily} kWh`);

  const kwh = Number(data.prediction_kwh).toFixed(2);
  setText("predictionResult", `${kwh} kWh`);
  setText("riskLevel", data.risk_level || "--");
  setText("estimatedCost", `₹ ${Number(data.estimated_cost_inr).toFixed(2)}`);
  setText("efficiencyScore", `${Number(data.sustainability_score)} / 100`);
  updateMeter(data.sustainability_score);
  updateRecommendations(data.recommendations);
  buildChartsFromAnalytics(analytics);
}

function buildCharts() {
  const data = window.dashboardData;
  if (!data || !window.Chart || isDashboardPage()) {
    return;
  }

  Chart.defaults.color = "#dce7ff";
  Chart.defaults.borderColor = "rgba(255,255,255,0.12)";

  const lineCanvas = document.getElementById("lineChart");
  if (lineCanvas) {
    new Chart(lineCanvas, {
      type: "line",
      data: {
        labels: data.charts.daily.labels,
        datasets: [
          {
            label: "kWh",
            data: data.charts.daily.values,
            borderColor: "#b7bdc8",
            backgroundColor: "rgba(183,189,200,0.22)",
            tension: 0.35,
            fill: true,
            pointRadius: 4,
            pointBackgroundColor: "#b7bdc8",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 1100, easing: "easeOutQuart" },
        plugins: { legend: { display: false } },
      },
    });
  }

  const barCanvas = document.getElementById("barChart");
  if (barCanvas) {
    buildDeviceBarChart(
      barCanvas,
      data.charts.device.labels,
      data.charts.device.values
    );
  }

  const pieCanvas = document.getElementById("pieChart");
  if (pieCanvas) {
    buildApplianceChart(
      pieCanvas,
      data.charts.appliance.labels,
      data.charts.appliance.values
    );
  }
}

function initPredictionFormPersistence(form) {
  if (isPageReload()) {
    clearSavedPredictionState();
  } else {
    restorePredictionForm(form);

    const raw = localStorage.getItem(PREDICTION_RESULT_KEY);
    if (raw && isDashboardPage()) {
      try {
        renderAnalytics(JSON.parse(raw));
      } catch {
        localStorage.removeItem(PREDICTION_RESULT_KEY);
      }
    } else if (raw) {
      restoreLegacyPredictionResult(JSON.parse(raw));
    }
  }

  FORM_FIELDS.forEach((field) => {
    const input = form.elements[field];
    if (input) {
      input.addEventListener("input", () => savePredictionForm(form));
      input.addEventListener("change", () => savePredictionForm(form));
    }
  });
}

function restoreLegacyPredictionResult(data) {
  const kwh = Number(data.prediction_kwh).toFixed(2);
  setText("predictionResult", `${kwh} kWh`);
  setText("predictedConsumption", `${kwh} kWh`);
  setText("riskLevel", data.risk_level || "--");
  setText("estimatedCost", `₹ ${Number(data.estimated_cost_inr).toFixed(2)}`);
  setText("efficiencyScore", `${Number(data.sustainability_score)} / 100`);
  updateMeter(data.sustainability_score);
  updateRecommendations(data.recommendations);
}

async function onPredict(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const output = document.getElementById("predictionResult");
  const submit = form.querySelector("button[type='submit']");
  const validationError = validateForm(form);

  hideValidationMessage();
  if (validationError) {
    showValidationMessage(validationError);
    return;
  }

  const payload = {};
  FORM_FIELDS.forEach((field) => {
    payload[field] = Number(form.elements[field].value);
  });

  try {
    submit.disabled = true;
    submit.textContent = "Generating prediction...";
    if (output) {
      output.textContent = "Generating prediction...";
    }

    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    savePredictionForm(form);
    savePredictionResult(data);

    if (isDashboardPage()) {
      renderAnalytics(data);
    } else {
      const kwh = Number(data.prediction_kwh).toFixed(2);
      if (output) {
        output.textContent = `${kwh} kWh`;
      }
      setText("predictedConsumption", `${kwh} kWh`);
      setText("riskLevel", data.risk_level || "--");
      setText("estimatedCost", `₹ ${Number(data.estimated_cost_inr).toFixed(2)}`);
      setText("efficiencyScore", `${Number(data.sustainability_score)} / 100`);
      updateMeter(data.sustainability_score);
      updateRecommendations(data.recommendations);
    }
  } catch (err) {
    showValidationMessage(err.message || "Prediction unavailable. Please try again.");
    if (output) {
      output.textContent = "-- kWh";
    }
  } finally {
    submit.disabled = false;
    submit.textContent = "Predict";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  if (isReportsPage()) {
    initReportsPage();
  } else if (!isDashboardPage()) {
    buildCharts();
  }

  const form = document.getElementById("predictionForm");
  if (form) {
    initPredictionFormPersistence(form);
    form.addEventListener("submit", onPredict);
  }
});
