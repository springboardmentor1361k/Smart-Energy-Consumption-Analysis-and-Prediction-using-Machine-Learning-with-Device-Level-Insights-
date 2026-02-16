/* ═══════════════════════════════════════════════════════════════════════
   SMART ENERGY DASHBOARD — JavaScript Controller
   ═══════════════════════════════════════════════════════════════════════ */

// ─── Globals ─────────────────────────────────────────────────────────
const charts = {};
const API = '';

// Chart.js global defaults
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyle = 'circle';
Chart.defaults.plugins.legend.labels.padding = 16;

// ─── Initialization ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    updateClock();
    setInterval(updateClock, 1000);
    loadOverview();
    loadTrend('hourly');
    loadHourlyPattern();
});

// ─── Clock ───────────────────────────────────────────────────────────
function updateClock() {
    const now = new Date();
    const el = document.getElementById('clock');
    if (el) {
        el.textContent = now.toLocaleString('en-IN', {
            year: 'numeric', month: 'short', day: 'numeric',
            hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true
        });
    }
}

// ─── Section Navigation ──────────────────────────────────────────────
function showSection(sectionId) {
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));

    const section = document.getElementById('section-' + sectionId);
    const btn = document.querySelector(`.nav-btn[data-section="${sectionId}"]`);
    if (section) section.classList.add('active');
    if (btn) btn.classList.add('active');

    // Lazy load section data
    const loaders = {
        'devices': loadDevices,
        'predictions': loadPredictions,
        'comparison': loadComparison,
        'suggestions': loadSuggestions,
        'visualizations': loadVisualizations,
    };
    if (loaders[sectionId]) loaders[sectionId]();
}

// ─── API Fetch Helper ────────────────────────────────────────────────
async function fetchAPI(endpoint) {
    try {
        const res = await fetch(API + endpoint);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (err) {
        console.error(`API Error (${endpoint}):`, err);
        return null;
    }
}

// ─── Create/Update Chart Helper ──────────────────────────────────────
function createChart(canvasId, config) {
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    charts[canvasId] = new Chart(ctx.getContext('2d'), config);
    return charts[canvasId];
}

// ═══════════════════════════════════════════════════════════════════════
// OVERVIEW SECTION
// ═══════════════════════════════════════════════════════════════════════

async function loadOverview() {
    const data = await fetchAPI('/api/overview');
    if (!data) return;

    setText('stat-records', data.total_records?.toLocaleString() || '--');
    setText('stat-avg-power', (data.avg_power_kw || 0).toFixed(4) + ' kW');
    setText('stat-accuracy', (data.lstm_accuracy || 0) + '%');
    setText('stat-monthly-cost', (data.currency || '₹') + (data.monthly_cost || 0).toLocaleString());
}

async function loadTrend(period) {
    updatePeriodButtons(event?.target, period);
    const data = await fetchAPI(`/api/consumption-trend?period=${period}`);
    if (!data || !data.chart) return;

    const chartData = data.chart;
    if (chartData.datasets?.length > 0) {
        chartData.datasets[0].borderColor = '#6366f1';
        chartData.datasets[0].backgroundColor = createGradient('trendChart', '#6366f1');
        chartData.datasets[0].fill = true;
        chartData.datasets[0].label = 'Global Active Power (kW)';
    }

    createChart('trendChart', {
        type: 'line',
        data: chartData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: { display: true, position: 'top' },
                tooltip: {
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(99, 102, 241, 0.3)',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 8,
                }
            },
            scales: {
                x: { display: true, grid: { display: false }, ticks: { maxTicksLimit: 12, maxRotation: 0 } },
                y: { display: true, grid: { color: 'rgba(255,255,255,0.04)' }, title: { display: true, text: 'Power (kW)' } }
            }
        }
    });
}

async function loadHourlyPattern() {
    const data = await fetchAPI('/api/hourly-pattern');
    if (!data) return;

    const datasets = [];
    const colors = { 'Global Active Power': '#6366f1', 'Kitchen': '#e74c3c', 'Laundry': '#3498db', 'HVAC': '#10b981' };

    for (const [label, values] of Object.entries(data.patterns || {})) {
        datasets.push({
            label: label,
            data: values,
            borderColor: colors[label] || '#94a3b8',
            backgroundColor: (colors[label] || '#94a3b8') + '22',
            fill: false,
            tension: 0.4,
            pointRadius: 3,
            pointHoverRadius: 6,
            borderWidth: 2,
        });
    }

    createChart('hourlyPatternChart', {
        type: 'line',
        data: { labels: data.labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: true, position: 'top' } },
            scales: {
                x: { grid: { display: false }, title: { display: true, text: 'Hour of Day' } },
                y: { grid: { color: 'rgba(255,255,255,0.04)' }, title: { display: true, text: 'Avg Power' } }
            }
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════
// DEVICES SECTION
// ═══════════════════════════════════════════════════════════════════════

let devicesLoaded = false;
async function loadDevices() {
    if (devicesLoaded) return;
    await loadDeviceChart('hourly');
    devicesLoaded = true;
}

async function loadDeviceChart(period) {
    updatePeriodButtons(event?.target, period);
    const data = await fetchAPI(`/api/device-consumption?period=${period}`);
    if (!data) return;

    // Device stat cards
    const statsGrid = document.getElementById('device-stats-cards');
    if (statsGrid && data.stats) {
        statsGrid.innerHTML = data.stats.map(d => `
            <div class="device-card" style="border-left: 4px solid ${d.color}">
                <div class="device-icon">${d.icon}</div>
                <div class="device-name">${d.name}</div>
                <div class="device-stat"><span class="device-stat-label">Share</span><span class="device-stat-value">${d.share}%</span></div>
                <div class="device-stat"><span class="device-stat-label">Mean</span><span class="device-stat-value">${d.mean?.toFixed(4)} kW</span></div>
                <div class="device-stat"><span class="device-stat-label">Max</span><span class="device-stat-value">${d.max?.toFixed(4)} kW</span></div>
                <div class="device-stat"><span class="device-stat-label">Peak Hour</span><span class="device-stat-value">${d.peak_hour}:00</span></div>
            </div>
        `).join('');
    }

    // Device line chart
    const chartData = data.chart;
    const deviceColors = ['#e74c3c', '#3498db', '#10b981'];
    chartData.datasets?.forEach((ds, i) => {
        ds.borderColor = deviceColors[i];
        ds.backgroundColor = deviceColors[i] + '22';
        ds.borderWidth = 2;
        ds.fill = true;
    });

    createChart('deviceChart', {
        type: 'line',
        data: chartData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: { legend: { display: true, position: 'top' } },
            scales: {
                x: { grid: { display: false }, ticks: { maxTicksLimit: 12, maxRotation: 0 } },
                y: { grid: { color: 'rgba(255,255,255,0.04)' }, title: { display: true, text: 'Power (Wh)' }, stacked: true }
            }
        }
    });

    // Pie chart
    if (data.stats) {
        createChart('devicePieChart', {
            type: 'doughnut',
            data: {
                labels: data.stats.map(d => d.name),
                datasets: [{
                    data: data.stats.map(d => d.share),
                    backgroundColor: data.stats.map(d => d.color),
                    borderColor: '#111827',
                    borderWidth: 3,
                    hoverOffset: 8,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '55%',
                plugins: {
                    legend: { position: 'bottom' },
                    tooltip: {
                        callbacks: {
                            label: ctx => `${ctx.label}: ${ctx.raw}%`
                        }
                    }
                }
            }
        });
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PREDICTIONS SECTION
// ═══════════════════════════════════════════════════════════════════════

let predictionsLoaded = false;
async function loadPredictions() {
    if (predictionsLoaded) return;
    const data = await fetchAPI('/api/predictions');
    if (!data) return;
    predictionsLoaded = true;

    // Metrics cards
    const metricsDiv = document.getElementById('prediction-metrics');
    if (metricsDiv && data.metrics) {
        const m = data.metrics;
        metricsDiv.innerHTML = `
            <div class="stat-card stat-card-gradient-3">
                <div class="stat-icon">🎯</div>
                <div class="stat-value">${m.accuracy}%</div>
                <div class="stat-label">Prediction Accuracy (R²)</div>
            </div>
            <div class="stat-card stat-card-gradient-2">
                <div class="stat-icon">📊</div>
                <div class="stat-value">${m.mae?.toFixed(6)}</div>
                <div class="stat-label">Mean Absolute Error</div>
            </div>
            <div class="stat-card stat-card-gradient-1">
                <div class="stat-icon">📈</div>
                <div class="stat-value">${m.mape}%</div>
                <div class="stat-label">MAPE</div>
            </div>
        `;
    }

    // Prediction chart
    if (data.chart) {
        const chartData = data.chart;
        if (chartData.datasets?.length >= 2) {
            chartData.datasets[0].borderColor = '#06b6d4';
            chartData.datasets[0].backgroundColor = '#06b6d422';
            chartData.datasets[0].label = 'Actual';
            chartData.datasets[0].borderWidth = 2;
            chartData.datasets[1].borderColor = '#10b981';
            chartData.datasets[1].backgroundColor = '#10b98122';
            chartData.datasets[1].label = 'LSTM Predicted';
            chartData.datasets[1].borderWidth = 2;
            chartData.datasets[1].borderDash = [5, 3];
        }

        createChart('predictionChart', {
            type: 'line',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { intersect: false, mode: 'index' },
                plugins: { legend: { display: true, position: 'top' } },
                scales: {
                    x: { grid: { display: false }, ticks: { maxTicksLimit: 15, maxRotation: 0 } },
                    y: { grid: { color: 'rgba(255,255,255,0.04)' }, title: { display: true, text: 'Power (kW)' } }
                }
            }
        });
    }

    // Error distribution
    if (data.error_distribution?.labels) {
        createChart('errorDistChart', {
            type: 'bar',
            data: {
                labels: data.error_distribution.labels,
                datasets: [{
                    label: 'Error Frequency',
                    data: data.error_distribution.values,
                    backgroundColor: '#6366f1aa',
                    borderColor: '#6366f1',
                    borderWidth: 1,
                    borderRadius: 4,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { display: false }, title: { display: true, text: 'Prediction Error' }, ticks: { maxTicksLimit: 10 } },
                    y: { grid: { color: 'rgba(255,255,255,0.04)' }, title: { display: true, text: 'Frequency' } }
                }
            }
        });
    }

    // Metrics table
    if (data.metrics) {
        const m = data.metrics;
        document.getElementById('metrics-table').innerHTML = `
            <table class="metrics-table">
                <thead>
                    <tr><th>Metric</th><th>Value</th></tr>
                </thead>
                <tbody>
                    <tr><td>MAE (kW)</td><td class="metric-good">${m.mae?.toFixed(6)}</td></tr>
                    <tr><td>RMSE (kW)</td><td class="metric-good">${m.rmse?.toFixed(6)}</td></tr>
                    <tr><td>R² Score</td><td class="metric-good">${m.r2?.toFixed(4)}</td></tr>
                    <tr><td>MAPE (%)</td><td class="metric-good">${m.mape}%</td></tr>
                    <tr><td>Accuracy</td><td class="metric-good">${m.accuracy}%</td></tr>
                    <tr><td>Total Predictions</td><td>${m.total_predictions?.toLocaleString()}</td></tr>
                </tbody>
            </table>
        `;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// MODEL COMPARISON SECTION
// ═══════════════════════════════════════════════════════════════════════

let comparisonLoaded = false;
async function loadComparison() {
    if (comparisonLoaded) return;
    const data = await fetchAPI('/api/model-comparison');
    if (!data) return;
    comparisonLoaded = true;

    const b = data.baseline;
    const l = data.lstm;

    // Model cards
    document.getElementById('baseline-card').innerHTML = `
        <h4 style="color: #ef4444;">📐 ${b.name}</h4>
        <div class="model-type">${b.type} Model</div>
        <div class="model-metric"><div class="model-metric-label">R² Score</div><div class="model-metric-value">${b.r2}</div></div>
        <div class="model-metric"><div class="model-metric-label">MAE</div><div class="model-metric-value">${b.mae}</div></div>
        <div class="model-metric"><div class="model-metric-label">RMSE</div><div class="model-metric-value">${b.rmse}</div></div>
        <div class="model-metric"><div class="model-metric-label">MAPE</div><div class="model-metric-value">${b.mape}%</div></div>
        <div class="model-metric"><div class="model-metric-label">Training Time</div><div class="model-metric-value">${b.training_time}</div></div>
    `;

    document.getElementById('lstm-card').innerHTML = `
        <h4 style="color: #10b981;">🧠 ${l.name}</h4>
        <div class="model-type">${l.type} Model</div>
        <div class="model-metric"><div class="model-metric-label">R² Score</div><div class="model-metric-value" style="color: #34d399;">${l.r2}</div></div>
        <div class="model-metric"><div class="model-metric-label">MAE</div><div class="model-metric-value" style="color: #34d399;">${l.mae}</div></div>
        <div class="model-metric"><div class="model-metric-label">RMSE</div><div class="model-metric-value" style="color: #34d399;">${l.rmse}</div></div>
        <div class="model-metric"><div class="model-metric-label">MAPE</div><div class="model-metric-value" style="color: #34d399;">${l.mape}%</div></div>
        <div class="model-metric"><div class="model-metric-label">Training Time</div><div class="model-metric-value">${l.training_time}</div></div>
    `;

    // Comparison bar chart (normalized for visibility)
    createChart('comparisonBarChart', {
        type: 'bar',
        data: {
            labels: ['R² Score', 'MAE (kW)', 'RMSE (kW)', 'MAPE (%)'],
            datasets: [
                {
                    label: 'Linear Regression',
                    data: [b.r2, b.mae, b.rmse, b.mape],
                    backgroundColor: '#ef4444aa',
                    borderColor: '#ef4444',
                    borderWidth: 2,
                    borderRadius: 6,
                },
                {
                    label: 'LSTM',
                    data: [l.r2, l.mae, l.rmse, l.mape],
                    backgroundColor: '#10b981aa',
                    borderColor: '#10b981',
                    borderWidth: 2,
                    borderRadius: 6,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'top' } },
            scales: {
                x: { grid: { display: false } },
                y: { grid: { color: 'rgba(255,255,255,0.04)' }, title: { display: true, text: 'Value' } }
            }
        }
    });

    // Improvement chart
    if (data.improvements) {
        const imp = data.improvements;
        createChart('improvementChart', {
            type: 'bar',
            data: {
                labels: Object.keys(imp).map(k => k.toUpperCase()),
                datasets: [{
                    label: 'Improvement %',
                    data: Object.values(imp),
                    backgroundColor: Object.values(imp).map(v => v > 0 ? '#10b981aa' : '#ef4444aa'),
                    borderColor: Object.values(imp).map(v => v > 0 ? '#10b981' : '#ef4444'),
                    borderWidth: 2,
                    borderRadius: 6,
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { label: ctx => `${ctx.raw > 0 ? '+' : ''}${ctx.raw}% improvement` } }
                },
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.04)' }, title: { display: true, text: 'Improvement (%)' } },
                    y: { grid: { display: false } }
                }
            }
        });
    }

    // Feature importance
    if (data.feature_importance?.length > 0) {
        const fi = data.feature_importance;
        const impKey = Object.keys(fi[0]).find(k => k !== 'Feature' && k !== 'feature');
        const featKey = Object.keys(fi[0]).find(k => k === 'Feature' || k === 'feature');

        if (impKey && featKey) {
            createChart('featureImportanceChart', {
                type: 'bar',
                data: {
                    labels: fi.map(f => f[featKey]).reverse(),
                    datasets: [{
                        label: 'Importance',
                        data: fi.map(f => Math.abs(f[impKey])).reverse(),
                        backgroundColor: '#8b5cf6aa',
                        borderColor: '#8b5cf6',
                        borderWidth: 1,
                        borderRadius: 4,
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { color: 'rgba(255,255,255,0.04)' }, title: { display: true, text: 'Absolute Coefficient' } },
                        y: { grid: { display: false }, ticks: { font: { size: 10 } } }
                    }
                }
            });
        }
    } else {
        document.getElementById('feature-importance-section').style.display = 'none';
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SMART SUGGESTIONS SECTION
// ═══════════════════════════════════════════════════════════════════════

let suggestionsLoaded = false;
async function loadSuggestions() {
    if (suggestionsLoaded) return;
    const data = await fetchAPI('/api/suggestions');
    if (!data) return;
    suggestionsLoaded = true;

    // Cost summary
    const costDiv = document.getElementById('cost-summary');
    if (costDiv && data.costs) {
        const c = data.costs;
        costDiv.innerHTML = `
            <div class="cost-item">
                <div class="cost-value">${c.currency || '₹'}${(c.daily_cost || 0).toFixed(1)}</div>
                <div class="cost-label">Daily Cost</div>
            </div>
            <div class="cost-item">
                <div class="cost-value">${c.currency || '₹'}${(c.monthly_cost || 0).toLocaleString()}</div>
                <div class="cost-label">Monthly Cost</div>
            </div>
            <div class="cost-item">
                <div class="cost-value">${c.currency || '₹'}${(c.annual_cost || 0).toLocaleString()}</div>
                <div class="cost-label">Annual Cost</div>
            </div>
            <div class="cost-item">
                <div class="cost-value">${(c.monthly_kwh || 0).toFixed(0)} kWh</div>
                <div class="cost-label">Monthly Usage</div>
            </div>
        `;
    }

    // Suggestion cards
    const grid = document.getElementById('suggestions-grid');
    if (grid && data.suggestions) {
        grid.innerHTML = data.suggestions.map(s => `
            <div class="suggestion-card" style="border-left-color: ${s.color || '#6366f1'}">
                <div class="suggestion-header">
                    <span class="suggestion-category">${s.category}</span>
                    <span class="suggestion-priority priority-${s.priority}">${s.priority}</span>
                </div>
                <div class="suggestion-title">${s.icon || '💡'} ${s.title}</div>
                <div class="suggestion-desc">${s.description}</div>
                <div class="suggestion-savings">💰 ${s.savings_potential}</div>
            </div>
        `).join('');
    }

    // Anomalies
    if (data.anomaly_count > 0) {
        loadAnomalies();
    }
}

async function loadAnomalies() {
    const data = await fetchAPI('/api/anomalies');
    if (!data || !data.anomalies?.length) return;

    const section = document.getElementById('anomaly-section');
    if (section) section.style.display = 'block';

    const list = document.getElementById('anomaly-list');
    if (list) {
        list.innerHTML = data.anomalies.slice(0, 10).map(a => `
            <div class="anomaly-item">
                <span>${a.message}</span>
                <span class="anomaly-severity severity-${a.severity}">${a.severity}</span>
            </div>
        `).join('');
    }
}

// ═══════════════════════════════════════════════════════════════════════
// VISUALIZATIONS SECTION
// ═══════════════════════════════════════════════════════════════════════

let vizLoaded = false;
async function loadVisualizations() {
    if (vizLoaded) return;
    const data = await fetchAPI('/api/visualizations');
    if (!data) return;
    vizLoaded = true;

    const gallery = document.getElementById('viz-gallery');
    if (gallery && data.visualizations) {
        gallery.innerHTML = data.visualizations.map(v => `
            <div class="viz-card">
                <a href="${v.url}" target="_blank">
                    <img src="${v.url}" alt="${v.title}" loading="lazy">
                </a>
                <div class="viz-title">${v.title}</div>
            </div>
        `).join('');
    }
}

// ─── Utility Functions ───────────────────────────────────────────────

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

function createGradient(canvasId, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return color + '33';
    const ctx = canvas.getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height || 300);
    gradient.addColorStop(0, color + '44');
    gradient.addColorStop(1, color + '05');
    return gradient;
}

function updatePeriodButtons(clickedBtn, period) {
    if (!clickedBtn) return;
    const parent = clickedBtn.closest('.period-selector');
    if (parent) {
        parent.querySelectorAll('.period-btn').forEach(b => b.classList.remove('active'));
        clickedBtn.classList.add('active');
    }
}
