document.addEventListener('DOMContentLoaded', function () {
    initDashboard();
    // Refresh every 10 seconds for simulation (slower for comparison)
    setInterval(updateDashboard, 10000);
});

let comparisonChart = null;
let deviceChart = null;
let trendChart = null;
let currentTrendType = 'hourly';
let trendData = null;

async function initDashboard() {
    await updateDashboard();
    await fetchSuggestions();
}

async function updateDashboard() {
    console.log("Updating dashboard data...");
    await fetchComparison();
    await fetchStats(); // For device breakdown
    await fetchTrends();
}

async function fetchTrends() {
    try {
        const response = await fetch('/api/trends');
        trendData = await response.json();
        renderTrendChart();
    } catch (error) {
        console.error('Error fetching trends:', error);
    }
}

function changeTrend(type) {
    currentTrendType = type;

    // Update active tab UI
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.innerText.toLowerCase() === type) {
            btn.classList.add('active');
        }
    });

    renderTrendChart();
}

async function fetchComparison() {
    try {
        const response = await fetch('/api/comparison');
        const data = await response.json();

        renderComparisonChart(data);

        // Update the prediction text based on the first future point
        const latestPred = data.lr[0];
        document.getElementById('prediction-text').innerText = `${latestPred.toFixed(2)} kW`;
        document.getElementById('model-text').innerText = `Comparison: Actual vs LR vs LSTM`;
    } catch (error) {
        console.error('Error fetching comparison:', error);
    }
}

async function fetchStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        renderDeviceChart(data);
    } catch (error) {
        console.error('Error fetching stats:', error);
    }
}

async function fetchSuggestions() {
    try {
        const response = await fetch('/api/suggestions');
        const suggestions = await response.json();

        const list = document.getElementById('suggestions-list');
        if (!list) return;
        list.innerHTML = '';

        suggestions.forEach(item => {
            const div = document.createElement('div');
            div.className = 'suggestion-item';
            div.innerHTML = `
                <h4>${item.title} </h4>
                <p>${item.description}</p>
            `;
            list.appendChild(div);
        });
    } catch (error) {
        console.error('Error fetching suggestions:', error);
    }
}

function renderTrendChart() {
    const ctx = document.getElementById('trendChart').getContext('2d');
    const data = trendData[currentTrendType];

    if (trendChart) {
        trendChart.data.labels = data.labels;
        trendChart.data.datasets[0].data = data.values;
        trendChart.data.datasets[0].label = `${currentTrendType.charAt(0).toUpperCase() + currentTrendType.slice(1)} Consumption (kW)`;
        trendChart.update();
        return;
    }

    trendChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.labels,
            datasets: [{
                label: `${currentTrendType.charAt(0).toUpperCase() + currentTrendType.slice(1)} Consumption (kW)`,
                data: data.values,
                backgroundColor: 'rgba(56, 189, 248, 0.5)',
                borderColor: '#38bdf8',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
            }
        }
    });
}

function renderComparisonChart(data) {
    const ctx = document.getElementById('comparisonChart').getContext('2d');

    if (comparisonChart) {
        comparisonChart.data.labels = data.timestamps;
        comparisonChart.data.datasets[0].data = data.actual;
        comparisonChart.data.datasets[1].data = data.lr;
        comparisonChart.data.datasets[2].data = data.lstm;
        comparisonChart.update('none');
        return;
    }

    comparisonChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.timestamps,
            datasets: [
                {
                    label: 'Actual',
                    data: data.actual,
                    borderColor: '#f8fafc',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.2
                },
                {
                    label: 'Baseline (LR)',
                    data: data.lr,
                    borderColor: '#38bdf8',
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.2
                },
                {
                    label: 'Predictive (LSTM)',
                    data: data.lstm,
                    borderColor: '#2ecc71',
                    fill: false,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top', labels: { color: '#94a3b8' } }
            },
            scales: {
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
            }
        }
    });
}

function renderDeviceChart(data) {
    const canvas = document.getElementById('deviceChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const avg1 = data.metering_1.reduce((a, b) => a + b, 0) / data.metering_1.length;
    const avg2 = data.metering_2.reduce((a, b) => a + b, 0) / data.metering_2.length;
    const avg3 = data.metering_3.reduce((a, b) => a + b, 0) / data.metering_3.length;

    if (deviceChart) {
        deviceChart.data.datasets[0].data = [avg1, avg2, avg3];
        deviceChart.update();
        return;
    }

    deviceChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Kitchen', 'Laundry', 'Climate Control'],
            datasets: [{
                data: [avg1, avg2, avg3],
                backgroundColor: ['#3498db', '#e74c3c', '#2ecc71'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'bottom', labels: { color: '#f8fafc' } }
            }
        }
    });
}
