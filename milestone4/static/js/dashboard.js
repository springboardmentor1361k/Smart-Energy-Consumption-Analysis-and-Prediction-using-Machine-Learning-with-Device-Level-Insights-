function setText(id, value) {
  const el = document.getElementById(id);
  if (el) {
    el.textContent = value;
  }
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
}

function buildCharts() {
  const data = window.dashboardData;
  if (!data || !window.Chart) {
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
    new Chart(barCanvas, {
      type: "bar",
      data: {
        labels: data.charts.device.labels,
        datasets: [
          {
            label: "Device Consumption (%)",
            data: data.charts.device.values,
            borderRadius: 10,
            backgroundColor: ["#27d17f", "#a9afb9", "#8d92a0", "#f8b84e", "#6f747d"],
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 1000, easing: "easeOutBack" },
        plugins: { legend: { display: false } },
      },
    });
  }

  const pieCanvas = document.getElementById("pieChart");
  if (pieCanvas) {
    new Chart(pieCanvas, {
      type: "pie",
      data: {
        labels: data.charts.appliance.labels,
        datasets: [
          {
            data: data.charts.appliance.values,
            backgroundColor: ["#27d17f", "#a9afb9", "#f8b84e", "#8d92a0", "#6f747d"],
            borderColor: "rgba(0,0,0,0)",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 1200, easing: "easeOutCubic" },
        plugins: { legend: { position: "bottom" } },
      },
    });
  }
}

async function onPredict(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const output = document.getElementById("predictionResult");
  const submit = form.querySelector("button[type='submit']");
  const formData = new FormData(form);

  const payload = {};
  formData.forEach((value, key) => {
    payload[key] = Number(value);
  });

  try {
    submit.disabled = true;
    submit.textContent = "Predicting...";
    output.textContent = "Calculating...";
    setText("riskLevel", "--");
    setText("estimatedCost", "₹ --");
    setText("efficiencyScore", "-- / 100");

    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    const kwh = Number(data.prediction_kwh).toFixed(2);
    output.textContent = `${kwh} kWh`;
    setText("predictedConsumption", `${kwh} kWh`);
    setText("riskLevel", data.risk_level || "--");
    setText("estimatedCost", `₹ ${Number(data.estimated_cost_inr).toFixed(2)}`);
    setText("efficiencyScore", `${Number(data.sustainability_score)} / 100`);
    updateMeter(data.sustainability_score);
    updateRecommendations(data.recommendations);
  } catch (err) {
    output.textContent = "Prediction unavailable";
    setText("riskLevel", "N/A");
    setText("estimatedCost", "₹ N/A");
    setText("efficiencyScore", "N/A");
  } finally {
    submit.disabled = false;
    submit.textContent = "Predict";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  buildCharts();
  const form = document.getElementById("predictionForm");
  if (form) {
    form.addEventListener("submit", onPredict);
  }
});
