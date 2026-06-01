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

const FORECAST_SCENARIOS = [
  {
    name: "Current Prediction",
    factor: 1.0,
    status: "Baseline forecast",
    description: "Projected from your latest dashboard prediction.",
  },
  {
    name: "Higher Usage Scenario",
    factor: 1.03,
    status: "Projected +3% usage",
    description:
      "Simulated increase with higher runtime across HVAC, laundry, lighting, and water heating.",
  },
  {
    name: "Peak Demand Scenario",
    factor: 1.06,
    status: "Projected peak load",
    description:
      "Simulated peak period with the largest share shifts to HVAC and water heating.",
  },
  {
    name: "Optimized Efficiency Scenario",
    factor: 0.94,
    status: "Projected savings",
    description:
      "Simulated savings with reduced HVAC, lighting, laundry, and water heating use.",
  },
];

// Device order: HVAC, Refrigerator, Washing Machine, Lighting, Water Heater
const SCENARIO_DEVICE_WEIGHT_MULT = [
  [1.0, 1.0, 1.0, 1.0, 1.0],
  [1.04, 1.0, 1.1, 1.08, 1.05],
  [1.15, 1.0, 1.06, 1.1, 1.18],
  [0.88, 1.0, 0.9, 0.82, 0.86],
];

const SCENARIO_EFFICIENCY_ADJUSTMENTS = [0, -3, -6, 6];
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

function buildHourlyLoadChart(canvas, labels, values) {
  const peakThreshold = Math.max(...values) * 0.85;
  const barColors = values.map((v) =>
    v >= peakThreshold ? "#f8b84e" : "#6ecf8f"
  );
  const hoverColors = values.map((v) =>
    v >= peakThreshold ? "#fcd34d" : "#8ee4aa"
  );

  return new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Load (kWh)",
          data: values,
          backgroundColor: barColors,
          hoverBackgroundColor: hoverColors,
          borderRadius: { topLeft: 10, topRight: 10, bottomLeft: 4, bottomRight: 4 },
          borderSkipped: false,
          maxBarThickness: 48,
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
          beginAtZero: true,
          border: { display: false },
          grid: { color: "rgba(255,255,255,0.06)" },
          ticks: {
            color: "#9eaccb",
            padding: 8,
            font: { size: 10, family: "Inter, sans-serif" },
            callback(value) {
              return `${value} kWh`;
            },
          },
        },
      },
    },
  });
}

function buildEfficiencyChart(canvas, labels, values) {
  const barColors = values.map((v) => {
    if (v >= 85) return "#6ecf8f";
    if (v >= 70) return "#f8b84e";
    return "#f87171";
  });

  return new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Efficiency",
          data: values,
          backgroundColor: barColors,
          borderRadius: { topRight: 10, bottomRight: 10, topLeft: 4, bottomLeft: 4 },
          borderSkipped: false,
          barThickness: 22,
        },
      ],
    },
    options: {
      indexAxis: "y",
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
              return ` ${context.parsed.x}% efficient`;
            },
          },
        },
      },
      scales: {
        x: {
          beginAtZero: true,
          max: 100,
          border: { display: false },
          grid: { color: "rgba(255,255,255,0.06)" },
          ticks: {
            color: "#9eaccb",
            padding: 8,
            font: { size: 10, family: "Inter, sans-serif" },
            callback(value) {
              return `${value}%`;
            },
          },
        },
        y: {
          grid: { display: false },
          border: { display: false },
          ticks: { color: "#eaf1ff", font: { size: 11, family: "Inter, sans-serif" } },
        },
      },
    },
  });
}

function isReportsPage() {
  return document.getElementById("scenarioSelector") !== null;
}

let reportsChartInstances = { line: null, bar: null };
let insightsChartInstances = { line: null, hourly: null };
let deviceAnalyticsChartInstances = { bar: null, efficiency: null };

function syncSecondaryPageVisibility(contentId, emptyId, hasData) {
  if (hasData) {
    hideElement(emptyId);
    showElement(contentId);
  } else {
    showElement(emptyId);
    hideElement(contentId);
  }
}

function destroyInsightsCharts() {
  insightsChartInstances.line?.destroy();
  insightsChartInstances.hourly?.destroy();
  insightsChartInstances = { line: null, hourly: null };
}

function destroyDeviceAnalyticsCharts() {
  deviceAnalyticsChartInstances.bar?.destroy();
  deviceAnalyticsChartInstances.efficiency?.destroy();
  deviceAnalyticsChartInstances = { bar: null, efficiency: null };
}

function destroyReportsCharts() {
  reportsChartInstances.line?.destroy();
  reportsChartInstances.bar?.destroy();
  reportsChartInstances = { line: null, bar: null };
}

function scenarioEfficiencyScore(baseEfficiency, index) {
  const adjustment = SCENARIO_EFFICIENCY_ADJUSTMENTS[index] || 0;
  return Math.max(0, Math.min(100, Math.round(baseEfficiency + adjustment)));
}

function normalizePercentages(values) {
  const total = values.reduce((sum, value) => sum + value, 0);
  const count = values.length;
  if (count === 0) {
    return [];
  }
  if (total <= 0) {
    const base = Math.floor(100 / count);
    const result = Array(count).fill(base);
    result[0] += 100 - result.reduce((sum, value) => sum + value, 0);
    return result;
  }

  const scaled = values.map((value) => (value / total) * 100);
  const rounded = scaled.map((value) => Math.round(value));
  const diff = 100 - rounded.reduce((sum, value) => sum + value, 0);
  if (diff) {
    const adjustIndex = scaled.indexOf(Math.max(...scaled));
    rounded[adjustIndex] += diff;
  }
  return rounded;
}

function scenarioDevicePercentages(basePercentages, scenarioIndex) {
  const multipliers =
    SCENARIO_DEVICE_WEIGHT_MULT[
      Math.min(scenarioIndex, SCENARIO_DEVICE_WEIGHT_MULT.length - 1)
    ];
  const weighted = basePercentages.map((value, index) =>
    Math.max(1, value * multipliers[index])
  );
  return normalizePercentages(weighted);
}

function setActiveScenarioUI(scenarioKey) {
  const reports = window.reportsData;
  const scenario = reports?.scenarios?.[scenarioKey];

  document.querySelectorAll(".scenario-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.scenario === scenarioKey);
  });
  document.querySelectorAll(".scenario-row").forEach((row) => {
    row.classList.toggle("active", row.dataset.scenario === scenarioKey);
  });
  setText("scenarioChartTitle", `Projected Daily Trend — ${scenarioKey}`);

  if (scenario) {
    setText("scenarioEfficiencyScore", `${scenario.efficiency_score}%`);
    setText("scenarioEfficiencyLabel", `Scenario Efficiency Score (${scenarioKey})`);
    setText("scenarioDescription", scenario.description || "--");
  }
}

function buildReportsScenarioCharts(scenarioKey) {
  const reports = window.reportsData;
  if (!reports?.scenarios?.[scenarioKey] || !window.Chart) {
    return;
  }

  const scenario = reports.scenarios[scenarioKey];
  destroyReportsCharts();
  setActiveScenarioUI(scenarioKey);

  Chart.defaults.color = "#dce7ff";
  Chart.defaults.borderColor = "rgba(255,255,255,0.12)";

  const lineCanvas = document.getElementById("lineChart");
  if (lineCanvas) {
    reportsChartInstances.line = buildLineChart(
      lineCanvas,
      scenario.daily.labels,
      scenario.daily.values
    );
  }

  const barCanvas = document.getElementById("barChart");
  if (barCanvas) {
    reportsChartInstances.bar = buildDeviceBarChart(
      barCanvas,
      scenario.device.labels,
      scenario.device.values
    );
  }
}

function initReportsPage() {
  const defaultScenario =
    window.reportsData?.default_scenario || FORECAST_SCENARIOS[0].name;
  buildReportsScenarioCharts(defaultScenario);

  document.querySelectorAll(".scenario-tab").forEach((tab) => {
    tab.addEventListener("click", () => buildReportsScenarioCharts(tab.dataset.scenario));
  });

  document.querySelectorAll(".scenario-row").forEach((row) => {
    row.addEventListener("click", () => buildReportsScenarioCharts(row.dataset.scenario));
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        buildReportsScenarioCharts(row.dataset.scenario);
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

function isInsightsPage() {
  return document.getElementById("hourlyChart") !== null;
}

function isDeviceAnalyticsPage() {
  return document.getElementById("efficiencyChart") !== null;
}

function clearSavedPredictionState() {
  localStorage.removeItem(PREDICTION_FORM_KEY);
  localStorage.removeItem(PREDICTION_RESULT_KEY);
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

function getStoredPrediction() {
  const raw = localStorage.getItem(PREDICTION_RESULT_KEY);
  if (!raw) {
    return null;
  }
  try {
    return JSON.parse(raw);
  } catch {
    localStorage.removeItem(PREDICTION_RESULT_KEY);
    return null;
  }
}

function toggleChartEmpty(chartId, isEmpty) {
  const canvas = document.getElementById(chartId);
  const empty = document.getElementById(`${chartId}Empty`);
  if (canvas) {
    canvas.classList.toggle("is-hidden", isEmpty);
  }
  if (empty) {
    empty.classList.toggle("is-hidden", !isEmpty);
  }
}

function showLoadingState() {
  hideElement("predictionCard");
  hideElement("analyticsSection");
  showElement("predictionLoading");
}

function hideLoadingState() {
  hideElement("predictionLoading");
}

function resetDashboard(form) {
  if (form) {
    form.reset();
  }
  clearSavedPredictionState();
  predictionData = null;
  destroyCharts();
  hideValidationMessage();
  hideLoadingState();
  hideElement("predictionCard");
  hideElement("analyticsSection");
  showElement("emptyState");

  toggleChartEmpty("lineChart", true);
  toggleChartEmpty("barChart", true);
  toggleChartEmpty("pieChart", true);

  setText("predictionResult", "-- kWh");
  setText("totalConsumption", "-- kWh");
  setText("predictedConsumption", "-- kWh");
  setText("peakUsageHours", "--");
  setText("peakUsageDetail", "--");
  setText("insightPeakDay", "--");
  setText("insightLowDay", "--");
  setText("insightAvgDaily", "-- kWh");
  setText("riskLevel", "--");
  setText("estimatedCost", "₹ --");
  setText("efficiencyScore", "-- / 100");
  setText("efficiencyLabel", "--%");
  setText("meterScoreLabel", "--%");
  setText("efficiencyRating", "--");
  updateMeter(0);
  updateRecommendations([]);
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

function updateMeter(score, rating) {
  const clamped = Math.max(0, Math.min(100, Number(score)));
  const meter = document.getElementById("meterFill");
  if (meter) {
    meter.style.width = `${clamped}%`;
  }
  setText("meterScoreLabel", `${clamped}%`);
  if (rating) {
    setText("efficiencyRating", rating);
  }
}

function destroyCharts() {
  Object.values(chartInstances).forEach((chart) => chart?.destroy());
  chartInstances = { line: null, bar: null, pie: null };
}

function buildChartsFromAnalytics(analytics) {
  if (!analytics || !window.Chart) {
    toggleChartEmpty("lineChart", true);
    toggleChartEmpty("barChart", true);
    toggleChartEmpty("pieChart", true);
    return;
  }

  const daily = analytics.charts?.daily;
  const device = analytics.charts?.device;
  const appliance = analytics.charts?.appliance;
  const hasDaily = daily?.values?.length > 0;
  const hasDevice = device?.values?.length > 0;
  const hasAppliance = appliance?.values?.length > 0;

  destroyCharts();
  Chart.defaults.color = "#dce7ff";
  Chart.defaults.borderColor = "rgba(255,255,255,0.12)";

  toggleChartEmpty("lineChart", !hasDaily);
  if (hasDaily) {
    const lineCanvas = document.getElementById("lineChart");
    chartInstances.line = buildLineChart(lineCanvas, daily.labels, daily.values);
  }

  toggleChartEmpty("barChart", !hasDevice);
  if (hasDevice) {
    const barCanvas = document.getElementById("barChart");
    chartInstances.bar = buildDeviceBarChart(
      barCanvas,
      device.labels,
      device.values
    );
  }

  toggleChartEmpty("pieChart", !hasAppliance);
  if (hasAppliance) {
    const pieCanvas = document.getElementById("pieChart");
    chartInstances.pie = buildApplianceChart(
      pieCanvas,
      appliance.labels,
      appliance.values
    );
  }
}

function renderAnalytics(data) {
  if (!data || !data.analytics) {
    return;
  }

  predictionData = data;
  hideLoadingState();
  const { analytics } = data;
  const cards = analytics.cards;
  const insights = analytics.insights;
  const peak = cards.peak_usage || {};

  hideElement("emptyState");
  showElement("predictionCard");

  const analyticsSection = document.getElementById("analyticsSection");
  showElement("analyticsSection");
  analyticsSection?.classList.add("analytics-reveal");

  document.getElementById("predictionCard")?.scrollIntoView({
    behavior: "smooth",
    block: "nearest",
  });

  setText("totalConsumption", `${cards.total_consumption} kWh`);
  setText("predictedConsumption", `${cards.predicted_next_day} kWh`);
  setText("peakUsageHours", peak.display || cards.peak_usage_hours || "--");
  setText(
    "peakUsageDetail",
    peak.peak_kwh ? `${peak.start} to ${peak.end} · ${peak.peak_kwh} kWh peak` : "--"
  );
  setText("insightPeakDay", `${insights.peak_day} (${insights.peak_value} kWh)`);
  setText("insightLowDay", `${insights.lowest_day} (${insights.lowest_value} kWh)`);
  setText("insightAvgDaily", `${insights.average_daily} kWh`);

  const kwh = Number(data.prediction_kwh).toFixed(2);
  setText("predictionResult", `${kwh} kWh`);
  setText("riskLevel", data.risk_level || "--");
  setText("estimatedCost", `₹ ${Number(data.estimated_cost_inr).toFixed(2)}`);
  setText("efficiencyScore", `${Number(data.sustainability_score)} / 100`);
  setText("efficiencyLabel", `${Number(data.sustainability_score)}%`);
  updateMeter(
    data.energy_efficiency_score ?? cards.energy_efficiency_score,
    data.efficiency_rating ?? cards.efficiency_rating
  );
  updateRecommendations(data.recommendations);
  buildChartsFromAnalytics(analytics);

  const tipButton = document.getElementById("sustainabilityTip");
  if (tipButton && analytics.metrics?.sustainability_formula) {
    tipButton.title = analytics.metrics.sustainability_formula;
  }
}

function hydrateInsightsPage(analytics) {
  const insights = analytics.insights || {};
  setText("insightPeakDayCard", `${insights.peak_day} (${insights.peak_value} kWh)`);
  setText("insightLowDayCard", `${insights.lowest_day} (${insights.lowest_value} kWh)`);
  setText("insightAvgDailyCard", `${insights.average_daily} kWh`);

  const list = document.getElementById("insightsList");
  const insightItems = Array.isArray(insights.insights) ? insights.insights : [];

  if (list) {
    list.innerHTML = "";
    if (insightItems.length) {
      insightItems.forEach((item) => {
        const li = document.createElement("li");
        li.textContent = item;
        list.appendChild(li);
      });
      hideElement("insightsListEmpty");
    } else {
      showElement("insightsListEmpty");
    }
  }

  syncSecondaryPageVisibility("insightsContent", "insightsEmpty", true);
  window.dashboardData = analytics;
  buildInsightsCharts();
}

function hydrateDeviceAnalyticsPage(analytics) {
  const device = analytics.charts.device;
  const efficiency = analytics.charts.device_efficiency;
  const maxValue = Math.max(...device.values);
  const maxIndex = device.values.indexOf(maxValue);
  const avgEfficiency = Math.round(
    efficiency.values.reduce((sum, value) => sum + value, 0) / efficiency.values.length
  );

  setText("deviceTopName", device.labels[maxIndex]);
  setText("deviceTopShare", `${maxValue}%`);
  setText("deviceAvgEfficiency", `${avgEfficiency}%`);

  const list = document.getElementById("deviceRecommendationsList");
  const recommendations = analytics.insights?.insights || [];

  if (list) {
    list.innerHTML = "";
    if (recommendations.length) {
      recommendations.forEach((item) => {
        const li = document.createElement("li");
        li.textContent = item;
        list.appendChild(li);
      });
      hideElement("deviceRecommendationsEmpty");
    } else {
      showElement("deviceRecommendationsEmpty");
    }
  }

  syncSecondaryPageVisibility("deviceAnalyticsContent", "deviceAnalyticsEmpty", true);
  window.dashboardData = analytics;
  buildDeviceAnalyticsCharts();
}

function hydrateReportsPage(data) {
  if (!data?.analytics) {
    return;
  }
  const analytics = data.analytics;
  const baseEfficiency = data.energy_efficiency_score || 0;
  const reportsData = {
    monthly_projection_kwh: Number((analytics.cards.predicted_next_day * 30).toFixed(2)),
    monthly_projection_cost: Number((analytics.cards.cost_estimation * 30).toFixed(2)),
    default_scenario: FORECAST_SCENARIOS[0].name,
    data_source: "projected",
    report_rows: [],
    scenarios: {},
  };

  FORECAST_SCENARIOS.forEach((scenario, index) => {
    const total = Number((analytics.cards.total_consumption * scenario.factor).toFixed(1));
    const cost = Number((total * 8 / 7).toFixed(2));
    const dailyValues = analytics.charts.daily.values.map((value) =>
      Number((value * scenario.factor).toFixed(1))
    );
    const deviceValues = scenarioDevicePercentages(analytics.charts.device.values, index);
    reportsData.scenarios[scenario.name] = {
      daily: { labels: analytics.charts.daily.labels, values: dailyValues },
      device: { labels: analytics.charts.device.labels, values: deviceValues },
      status: scenario.status,
      description: scenario.description,
      factor: scenario.factor,
      efficiency_score: scenarioEfficiencyScore(baseEfficiency, index),
      is_projected: true,
    };
    reportsData.report_rows.push([scenario.name, total, cost, scenario.status]);
  });

  window.reportsData = reportsData;
  syncSecondaryPageVisibility("reportsContent", "reportsEmpty", true);
  setText("monthlyProjectionKwh", `${reportsData.monthly_projection_kwh} kWh`);
  setText("monthlyProjectionCost", `₹${reportsData.monthly_projection_cost}`);

  const tableBody = document.getElementById("reportTableBody");
  const scenarioSelector = document.getElementById("scenarioSelector");
  if (tableBody) {
    tableBody.innerHTML = "";
    reportsData.report_rows.forEach((row) => {
      const tr = document.createElement("tr");
      tr.className = "scenario-row";
      tr.dataset.scenario = row[0];
      tr.tabIndex = 0;
      tr.setAttribute("role", "button");
      tr.innerHTML = `<td>${row[0]}</td><td>${row[1]}</td><td>${row[2]}</td><td>${row[3]}</td>`;
      tableBody.appendChild(tr);
    });
  }
  if (scenarioSelector) {
    scenarioSelector.innerHTML = "";
    reportsData.report_rows.forEach((row) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "scenario-tab";
      button.dataset.scenario = row[0];
      button.textContent = row[0];
      scenarioSelector.appendChild(button);
    });
  }

  initReportsPage();
}

function initSecondaryPagesFromPrediction() {
  const stored = getStoredPrediction();
  if (!stored?.analytics) {
    return;
  }

  if (isInsightsPage()) {
    hydrateInsightsPage(stored.analytics);
  } else if (isDeviceAnalyticsPage()) {
    hydrateDeviceAnalyticsPage(stored.analytics);
  } else if (isReportsPage()) {
    hydrateReportsPage(stored);
  }
}

function buildInsightsCharts() {
  const data = window.dashboardData;
  destroyInsightsCharts();

  if (!data || !window.Chart) {
    toggleChartEmpty("lineChart", true);
    toggleChartEmpty("hourlyChart", true);
    return;
  }

  const hasDaily = data.charts?.daily?.values?.length > 0;
  const hasHourly = data.charts?.hourly?.values?.length > 0;

  Chart.defaults.color = "#dce7ff";
  Chart.defaults.borderColor = "rgba(255,255,255,0.12)";

  const lineCanvas = document.getElementById("lineChart");
  if (lineCanvas) {
    if (hasDaily) {
      toggleChartEmpty("lineChart", false);
      insightsChartInstances.line = buildLineChart(
        lineCanvas,
        data.charts.daily.labels,
        data.charts.daily.values
      );
    } else {
      toggleChartEmpty("lineChart", true);
    }
  }

  const hourlyCanvas = document.getElementById("hourlyChart");
  if (hourlyCanvas) {
    if (hasHourly && data.charts.hourly) {
      toggleChartEmpty("hourlyChart", false);
      insightsChartInstances.hourly = buildHourlyLoadChart(
        hourlyCanvas,
        data.charts.hourly.labels,
        data.charts.hourly.values
      );
    } else {
      toggleChartEmpty("hourlyChart", true);
    }
  }
}

function buildDeviceAnalyticsCharts() {
  const data = window.dashboardData;
  destroyDeviceAnalyticsCharts();

  if (!data || !window.Chart) {
    toggleChartEmpty("barChart", true);
    toggleChartEmpty("efficiencyChart", true);
    return;
  }

  const hasDevice = data.charts?.device?.values?.length > 0;
  const hasEfficiency = data.charts?.device_efficiency?.values?.length > 0;

  Chart.defaults.color = "#dce7ff";
  Chart.defaults.borderColor = "rgba(255,255,255,0.12)";

  const barCanvas = document.getElementById("barChart");
  if (barCanvas) {
    if (hasDevice) {
      toggleChartEmpty("barChart", false);
      deviceAnalyticsChartInstances.bar = buildDeviceBarChart(
        barCanvas,
        data.charts.device.labels,
        data.charts.device.values
      );
    } else {
      toggleChartEmpty("barChart", true);
    }
  }

  const efficiencyCanvas = document.getElementById("efficiencyChart");
  if (efficiencyCanvas) {
    if (hasEfficiency && data.charts.device_efficiency) {
      toggleChartEmpty("efficiencyChart", false);
      deviceAnalyticsChartInstances.efficiency = buildEfficiencyChart(
        efficiencyCanvas,
        data.charts.device_efficiency.labels,
        data.charts.device_efficiency.values
      );
    } else {
      toggleChartEmpty("efficiencyChart", true);
    }
  }
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
    buildLineChart(lineCanvas, data.charts.daily.labels, data.charts.daily.values);
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

function initFormEnterNavigation(form) {
  if (!form) {
    return;
  }

  FORM_FIELDS.forEach((field, index) => {
    const input = form.elements[field];
    if (!input) {
      return;
    }

    input.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") {
        return;
      }

      event.preventDefault();

      const nextField = FORM_FIELDS[index + 1];
      if (nextField) {
        const nextInput = form.elements[nextField];
        if (nextInput) {
          nextInput.focus();
          if (typeof nextInput.select === "function") {
            nextInput.select();
          }
        }
        return;
      }

      const submit = form.querySelector("button[type='submit']");
      if (submit) {
        form.requestSubmit(submit);
      }
    });
  });
}

function initPredictionFormPersistence(form) {
  restorePredictionForm(form);

  const stored = getStoredPrediction();
  if (stored?.analytics && isDashboardPage()) {
    renderAnalytics(stored);
  } else if (stored) {
    restoreLegacyPredictionResult(stored);
  }

  FORM_FIELDS.forEach((field) => {
    const input = form.elements[field];
    if (input) {
      input.addEventListener("input", () => savePredictionForm(form));
      input.addEventListener("change", () => savePredictionForm(form));
    }
  });

  initFormEnterNavigation(form);
}

function restoreLegacyPredictionResult(data) {
  const kwh = Number(data.prediction_kwh).toFixed(2);
  setText("predictionResult", `${kwh} kWh`);
  setText("predictedConsumption", `${kwh} kWh`);
  setText("riskLevel", data.risk_level || "--");
  setText("estimatedCost", `₹ ${Number(data.estimated_cost_inr).toFixed(2)}`);
  setText("efficiencyScore", `${Number(data.sustainability_score)} / 100`);
  updateMeter(
    data.energy_efficiency_score ?? data.sustainability_score,
    data.efficiency_rating || "Average"
  );
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
    submit.textContent = "Predicting...";
    showLoadingState();

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
      hideLoadingState();
      const kwh = Number(data.prediction_kwh).toFixed(2);
      if (output) {
        output.textContent = `${kwh} kWh`;
      }
      setText("predictedConsumption", `${kwh} kWh`);
      setText("riskLevel", data.risk_level || "--");
      setText("estimatedCost", `₹ ${Number(data.estimated_cost_inr).toFixed(2)}`);
      setText("efficiencyScore", `${Number(data.sustainability_score)} / 100`);
      updateMeter(
        data.energy_efficiency_score ?? data.sustainability_score,
        data.efficiency_rating || "Average"
      );
      updateRecommendations(data.recommendations);
    }
  } catch (err) {
    hideLoadingState();
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
  if (isDashboardPage()) {
    toggleChartEmpty("lineChart", true);
    toggleChartEmpty("barChart", true);
    toggleChartEmpty("pieChart", true);
  } else if (isReportsPage()) {
    initSecondaryPagesFromPrediction();
    syncSecondaryPageVisibility(
      "reportsContent",
      "reportsEmpty",
      Boolean(getStoredPrediction()?.analytics)
    );
  } else if (isInsightsPage()) {
    initSecondaryPagesFromPrediction();
    syncSecondaryPageVisibility(
      "insightsContent",
      "insightsEmpty",
      Boolean(getStoredPrediction()?.analytics)
    );
  } else if (isDeviceAnalyticsPage()) {
    initSecondaryPagesFromPrediction();
    syncSecondaryPageVisibility(
      "deviceAnalyticsContent",
      "deviceAnalyticsEmpty",
      Boolean(getStoredPrediction()?.analytics)
    );
  } else if (!isDashboardPage()) {
    buildCharts();
  }

  const form = document.getElementById("predictionForm");
  if (form) {
    initPredictionFormPersistence(form);
    form.addEventListener("submit", onPredict);
  }

  const resetBtn = document.getElementById("resetBtn");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => resetDashboard(form));
  }
});
